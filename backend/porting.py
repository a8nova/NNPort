import time
import os
import json
import uuid
import shutil
import subprocess
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from backend.targets import TargetType, DeviceConfig
from backend.core import ComparisonEngine
from backend.project_ops import apply_ops_in_place, OpsValidationError, validate_ops_object
from backend.project_analyzer import analyze_project, write_profile
from backend.build_planner import generate_cmake_harness
from backend.build_executor import build_cmake_harness, BuildError, elf_has_main
from backend.nnport_workspace import get_nnport_workspace
from backend.project_promoter import promotion_ops
from backend.failure_classifier import (
    normalize_signature,
    classify_compilation_error,
    classify_runtime_error,
    FailureClass,
)
import asyncio

import google.generativeai as genai
import openai
from dotenv import load_dotenv

# Load environment variables from .env.dev if not already loaded
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.dev')
if os.path.exists(env_path) and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    load_dotenv(env_path)

class CodeGenerator:
    def __init__(self):
        # Default to OpenAI as requested
        self.provider = "openai"
        self.model = None
        self.openai_client = None
        self.openai_legacy = False
        
        # Check OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                # Try new OpenAI SDK (v1.0+)
                self.openai_client = openai.OpenAI(api_key=openai_key)
                self.openai_legacy = False
            except AttributeError:
                # Fall back to old OpenAI SDK (v0.x)
                openai.api_key = openai_key
                self.openai_legacy = True
                print("⚠ Using legacy OpenAI SDK (v0.x). Consider upgrading: pip install --upgrade openai")
        
        # Check Gemini (Legacy/Fallback)
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
             genai.configure(api_key=gemini_key)
             self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        else:
             self.gemini_model = None

    def generate(
        self,
        original_model_source: str,
        target: TargetType,
        iteration: int,
        error_feedback: str = None,
        debug_instructions: str = "",
        device_config: DeviceConfig = None,
        project_context: dict = None,
        output_format: str = "code",
        max_json_retries: int = 3,
    ) -> Any:
        """
        Generates code for the target architecture.
        Prioritizes OpenAI (GPT-4o), falls back to Gemini, then Mock.
        Supports full project context for multi-file GPU projects.

        output_format:
          - "code": legacy behavior, returns {"host_code": str, "kernel_code": Optional[str]}
          - "json_ops": returns parsed JSON object {"ops": [...]} for in-place project editing
        """

        def _strip_to_json(text: str) -> str:
            """Best-effort extraction of a single JSON object from LLM output."""
            text = self._strip_markdown(text or "")
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
            return text.strip()

        def _validate_minimal_ops(obj: Any) -> Optional[str]:
            """Lightweight validation; full validation is handled elsewhere."""
            if not isinstance(obj, dict):
                return "Top-level JSON must be an object."
            ops = obj.get("ops")
            if not isinstance(ops, list) or not ops:
                return 'Top-level JSON must contain non-empty key "ops": [...].'
            for i, op in enumerate(ops):
                if not isinstance(op, dict):
                    return f"ops[{i}] must be an object."
                if op.get("type") not in ("create", "update", "delete", "rename"):
                    return f'ops[{i}].type must be one of create/update/delete/rename.'
            return None
        
        error_context = ""
        if error_feedback:
            error_context = f"""
PREVIOUS ATTEMPT FAILED:
{error_feedback}

Please fix this issue in your new code generation.
"""
        
        user_guidance = ""
        if debug_instructions:
            user_guidance = f"""
USER DEBUGGING GUIDANCE:
{debug_instructions}

Take this guidance into account when generating or fixing the code.
"""
        
        project_info = ""
        if project_context:
            files_list = "\n".join([f"  - {path}" for path in project_context["files"].keys()])
            project_info = f"""
PROJECT CONTEXT:
This code is part of a multi-file GPU project: {project_context["folder_name"]}

Files in project:
{files_list}

You have access to all these files. When fixing errors, consider:
1. Dependencies between files (headers, includes)
2. Shared data structures and types
3. Function definitions across files
4. Build order and linking requirements

Main entry point: {project_context.get("main_file", "unknown")}

Full file contents:
"""
            for path, content in project_context["files"].items():
                project_info += f"\n--- {path} ---\n{content}\n"
        
        device_context = ""
        if device_config and target in [TargetType.OPENCL, TargetType.CUDA]:
            if device_config.compute_backend != "auto":
                device_context = f"""
COMPUTE DEVICE SELECTION:
- Backend: {device_config.compute_backend.upper()}
- Device Type: {device_config.compute_device_type}
- Platform ID: {device_config.compute_platform_id}
- Device ID: {device_config.compute_device_id}

When generating OpenCL/CUDA code:
1. Use CL_DEVICE_TYPE_{device_config.compute_device_type.upper()} when getting devices
2. Select platform {device_config.compute_platform_id} and device {device_config.compute_device_id}
3. Include proper device selection in the runtime code
"""
        
        json_ops_contract = ""
        if output_format == "json_ops":
            json_ops_contract = """
OUTPUT FORMAT (STRICT):
Return ONLY a single JSON object (no markdown, no prose) of the exact shape:
{
  "ops": [
    {
      "type": "create" | "update" | "delete" | "rename",
      "path": "relative/path/from/project/root",              // for create/update/delete
      "content": "full file contents",                        // for create/update only
      "expected_sha256_before": "optional hex sha256",         // optional for update/delete
      "from_path": "old/path", "to_path": "new/path"           // for rename only
    }
  ]
}

Rules:
- Use ONLY relative paths, never absolute paths.
- Never use path traversal like ../
- For updates, prefer changing as few files as possible, but ALWAYS provide full file contents for each updated/created file.
- Ensure the resulting project compiles and runs for the specified target.
"""

        # Different prompts for OpenCL vs other targets
        if target == TargetType.OPENCL:
            prompt = f"""
You are an expert AI compiler engineer specialized in porting PyTorch models to OpenCL for GPU execution.
Your task is to port the following PyTorch model to OpenCL, generating TWO separate files:

Context:
- Iteration: {iteration} (0=Initial Draft, 1=Refinement, 2=Optimization/Final)
- Source Model:
```python
{original_model_source}
```
{error_context}
{user_guidance}
{project_info}
{device_context}

Requirements:
1. Generate TWO separate code blocks marked with "/* HOST_CODE */" and "/* KERNEL_CODE */"
2. HOST CODE (C++):
   - Include <CL/cl.h> and OpenCL API calls
   - Use ONLY C standard library: <stdio.h>, <stdlib.h>, <string.h>, <math.h>
   - DO NOT use <iostream>, <vector>, <string> or other C++ STL headers
   - Use printf() for output, NOT cout
   - Use malloc/free, NOT new/delete
   - Read input from "input.bin" file (float32 binary format)
   - Load kernel source from "kernel.cl" file using fopen/fread
   - Set up OpenCL context, queue, buffers, and kernel
   - Execute kernel on GPU
   - Read results back from GPU
   - Write COMPLETE output array to "output.bin" using fwrite()
   - CRITICAL: Print detailed GPU initialization logs using printf():
     * Print number of platforms found
     * Print platform name/vendor for selected platform
     * Print number of devices found
     * Print device name/type for selected device
     * Print "Building kernel..." before clBuildProgram
     * Print "Kernel compiled successfully" or build errors after clBuildProgram
     * Print "Executing kernel on GPU..." before clEnqueueNDRangeKernel
     * Print "Execution complete" after clFinish
     * Print final output values
   - Add error checking and print OpenCL error codes if any operation fails
3. KERNEL CODE (OpenCL C):
   - Pure OpenCL kernel functions only
   - Use __kernel, __global, get_global_id(), etc.
   - Implement the actual computation that runs on GPU
4. File loading pattern for host code:
```cpp
FILE* fp = fopen("kernel.cl", "r");
fseek(fp, 0, SEEK_END);
size_t size = ftell(fp);
rewind(fp);
char* kernel_source = (char*)malloc(size + 1);
fread(kernel_source, 1, size, fp);
kernel_source[size] = '\\0';
fclose(fp);
```

Generate the code now in this format:
/* HOST_CODE */
<C++ host code here>
/* KERNEL_CODE */
<OpenCL kernel code here>

Do not wrap in markdown code blocks, just return raw code.
"""
        else:
            prompt = f"""
You are an expert AI compiler engineer specialized in porting PyTorch models to high-performance hardware targets.
Your task is to port the following PyTorch model content to {target.value}.

Context:
- Iteration: {iteration} (0=Initial Draft, 1=Refinement, 2=Optimization/Final)
- Source Model:
```python
{original_model_source}
```
{error_context}
{user_guidance}
{device_context}
Requirements:
1. Output ONLY valid, compilable C source code (C11) that can be compiled with gcc or Android NDK clang.
2. Do NOT use C++ features at all - no classes, cout, vector, string, new/delete, R"..." raw strings.
3. Do NOT include OpenCL/CUDA headers like <CL/cl.h> or <cuda.h>. The code must compile standalone.
4. Only use these headers: <stdio.h>, <stdlib.h>, <string.h>, <math.h>
5. Use FIXED-SIZE arrays or malloc/free for dynamic arrays. Do NOT use VLAs (Variable Length Arrays) with initializers.
6. Include a main() function that:
   - Reads input from "input.bin" file (float32 binary format) instead of using hardcoded test values
   - Performs the inference computation
   - Prints results using printf()
   - Writes the COMPLETE output array to "output.bin" using fwrite()
   - CRITICAL: Write ALL output values (the entire array) in one fwrite() call: fwrite(output_array, sizeof(float), num_output_elements, fp)
7. For arrays: Either use fixed sizes like float weights[5][10] or use malloc.
8. Use regular C string literals "..." NEVER raw strings like R"CL(...)CL" or R"EOF(...)EOF".
9. Keep it simple - pure procedural C code only. No C++ syntax whatsoever.
10. IMPORTANT: Read input from "input.bin" file, do NOT use hardcoded test data!

IMPORTANT: Do NOT use:
  ❌ const char* x = R"CL(...)CL";              // Raw string - C++11 only
  ❌ float arr[size] = {{...}};                   // VLA with initializer - NOT allowed in C
  ❌ float input[10] = {{0.1, 0.2, ...}};         // Hardcoded test data - NOT allowed
  ✅ float arr[5] = {{...}};                      // Fixed size with initializer - OK for weights
  ✅ float* arr = malloc(size * sizeof(float)); // Dynamic allocation - OK

EXAMPLE: Reading input from file:
```c
FILE* fp = fopen("input.bin", "rb");
float input[NUM_INPUTS];
fread(input, sizeof(float), NUM_INPUTS, fp);
fclose(fp);
```

Generate the code now. Do not wrap in markdown code blocks, just return raw code if possible, or I will strip them.
"""     

        if output_format == "json_ops":
            # Wrap the code-gen instructions into an edit-plan request (LLM must emit JSON ops only).
            prompt = f"""
You are an expert embedded + systems engineer. You must modify the given project to fix the failure and make it build/run.

{json_ops_contract}

PROJECT GOAL:
- Target: {target.value}
- Iteration: {iteration}

INPUT (reference model / intent):
```python
{original_model_source}
```
{error_context}
{user_guidance}
{project_info}
{device_context}

IMPLEMENTATION NOTES:
- If you need to add helper headers/sources, create them under the project root.
- If this is an OpenCL target for Android, do NOT assume the NDK provides OpenCL headers.
- Prefer including a project-local header like \"nnport_opencl.h\" instead of <CL/cl.h> if needed.

Return the JSON now.
"""

        def _call_openai(p: str) -> Optional[str]:
            if not (self.openai_client or self.openai_legacy):
                return None
            if self.openai_legacy:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a specialized code generation assistant."},
                        {"role": "user", "content": p},
                    ],
                )
                return response.choices[0].message.content
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a specialized code generation assistant."},
                    {"role": "user", "content": p},
                ],
            )
            return response.choices[0].message.content

        def _call_gemini(p: str) -> Optional[str]:
            if not self.gemini_model:
                return None
            response = self.gemini_model.generate_content(p)
            return response.text

        # Try OpenAI
        if output_format == "json_ops":
            last_err: Optional[str] = None
            for attempt in range(max_json_retries):
                attempt_prompt = prompt if attempt == 0 else f"{prompt}\n\nJSON VALIDATION ERROR:\n{last_err}\n\nReturn corrected JSON only."
                # Prefer OpenAI, fallback to Gemini
                raw = None
                try:
                    raw = _call_openai(attempt_prompt)
                except Exception as e:
                    print(f"OpenAI Error: {e}")
                if raw is None:
                    try:
                        raw = _call_gemini(attempt_prompt)
                    except Exception as e:
                        print(f"Gemini Error: {e}")
                if not raw:
                    last_err = "Empty model response."
                    continue
                try:
                    json_text = _strip_to_json(raw)
                    obj = json.loads(json_text)
                except Exception as e:
                    last_err = f"Could not parse JSON: {e}"
                    continue
                v_err = _validate_minimal_ops(obj)
                if v_err:
                    last_err = v_err
                    continue
                return obj
            raise Exception(f"LLM did not return valid JSON ops after {max_json_retries} attempts: {last_err}")

        # Legacy: code blob generation
        code_text = None
        try:
            code_text = _call_openai(prompt)
        except Exception as e:
            print(f"OpenAI Error: {e}")
        if not code_text:
            try:
                code_text = _call_gemini(prompt)
            except Exception as e:
                print(f"Gemini Error: {e}")

        if code_text:
            code_text = self._sanitize_code(code_text, target)
            if target == TargetType.OPENCL:
                return self._parse_opencl_response(code_text)
            return {"host_code": code_text, "kernel_code": None}

        # Fallback mock
        mock_code = f"// ERROR: No API Keys available (OpenAI or Gemini).\n// Please set OPENAI_API_KEY to use AI code generation.\n// Mocking fallback for now...\n" + self._get_mock_template(original_model_source, target, iteration)
        return {"host_code": mock_code, "kernel_code": None}
    
    def _parse_opencl_response(self, code: str) -> Dict[str, str]:
        """Parse OpenCL response into host_code and kernel_code"""
        # Look for markers
        if '/* HOST_CODE */' in code and '/* KERNEL_CODE */' in code:
            parts = code.split('/* KERNEL_CODE */')
            host_part = parts[0].replace('/* HOST_CODE */', '').strip()
            kernel_part = parts[1].strip()
            return {
                "host_code": host_part,
                "kernel_code": kernel_part
            }
        else:
            # Fallback: treat entire code as host code
            print("Warning: Could not find OpenCL markers, treating as single file")
            return {
                "host_code": code,
                "kernel_code": None
            }
    
    def _sanitize_code(self, code: str, target: TargetType = None) -> str:
        """Remove problematic includes and C++ syntax that would prevent compilation"""
        import re
        
        # First strip markdown
        code = self._strip_markdown(code)
        
        # For OpenCL target, don't sanitize - allow OpenCL headers and C++ features
        if target == TargetType.OPENCL:
            return code
        
        original_code = code
        
        # AGGRESSIVELY remove raw string literals using regex - they span multiple lines
        # Match R"delimiter(...any content...)delimiter" where delimiter can be CL, EOF, etc.
        code = re.sub(r'R"([A-Z]*)\(.*?\)\1"', '""  /* raw string removed */', code, flags=re.DOTALL)
        
        if 'R"' in original_code:
            print(f"Sanitizer: Detected and removing raw string literals from generated code")
        
        # Also remove any lines that still have R" syntax
        lines = code.split('\n')
        sanitized = []
        skip_mode = False
        
        for line in lines:
            # If we see R" at the start of a line, enter skip mode
            if 'R"' in line:
                skip_mode = True
                sanitized.append(f"// {line.strip()}  // REMOVED: Raw string literal (C++ only)")
                continue
            
            # If in skip mode, keep skipping until we see the end
            if skip_mode:
                if '";' in line or ')CL";' in line or ')EOF";' in line:
                    skip_mode = False
                    sanitized.append(f"// {line.strip()}  // REMOVED: End of raw string")
                else:
                    sanitized.append(f"// {line.strip()}  // REMOVED: Raw string content")
                continue
            
            # Skip problematic includes
            if any(x in line for x in ['#include <CL/', '#include <cuda', '#include <cl_', 
                                        '#include <iostream>', '#include <vector>', '#include <string>']):
                sanitized.append(f"// {line.strip()}  // REMOVED: Unsupported include")
            else:
                sanitized.append(line)
        
        return '\n'.join(sanitized)
    
    def _strip_markdown(self, code: str) -> str:
        """Remove markdown code block markers and explanatory text from generated code"""
        import re
        
        # Remove opening code fence (```c, ```cpp, ```C, etc.)
        code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        
        # Remove closing code fence
        code = re.sub(r'\n```\s*$', '', code)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any stray ``` markers in the middle
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Skip lines that are just ```
            if stripped == '```' or stripped.startswith('```') and len(stripped) < 10:
                continue
            lines.append(line)
        
        code = '\n'.join(lines)
        
        # CRITICAL: Remove AI explanatory text before/after code
        # Look for the first real C/C++ code marker (usually #include or /*)
        code_start_patterns = [r'^\s*#include', r'^\s*#define', r'^\s*/\*', r'^\s*//', r'^\s*typedef', r'^\s*struct', r'^\s*int\s+main', r'^\s*void\s+', r'^\s*float\s+']
        
        lines = code.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find where actual code starts (skip AI explanations like "Here is...")
        for i, line in enumerate(lines):
            if any(re.match(pattern, line) for pattern in code_start_patterns):
                start_idx = i
                break
        
        # Find where actual code ends (before explanations like "This code does...")
        # Heuristic: after the last closing brace at column 0, ignore remaining text
        last_brace_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '}' or lines[i].startswith('}'):
                last_brace_idx = i
                break
        
        if last_brace_idx > start_idx:
            # Check if there's substantial text after the last brace
            remaining_lines = lines[last_brace_idx + 1:]
            non_empty = [l for l in remaining_lines if l.strip() and not l.strip().startswith('//')]
            
            # If there are more than 2 non-empty lines after last brace, likely explanatory text
            if len(non_empty) > 2:
                # Check if these lines look like natural language (contain common words)
                explanation_keywords = ['this', 'the', 'code', 'implementation', 'performs', 'using', 'please', 'ensure', 'note']
                if any(any(keyword in line.lower() for keyword in explanation_keywords) for line in non_empty[:3]):
                    end_idx = last_brace_idx + 1
                    print(f"Stripped explanatory text after code (removed {len(non_empty)} lines)")
        
        return '\n'.join(lines[start_idx:end_idx])

    def _get_mock_template(self, original_model_source: str, target: TargetType, iteration: int) -> str:
        if target == TargetType.OPENCL:
            # OpenCL implementation in pure C
            code_template = f"""
// Generated C Code for OpenCL - Iteration {iteration}
#include <stdio.h>
#include <stdlib.h>

void run_inference(float* input, float* output, int size) {{
    // Simple element-wise operation
    for(int i = 0; i < size; i++) {{
        output[i] = input[i] * {0.5 if iteration < 2 else 1.0}f;
    }}
}}

int main() {{
    printf("Inference complete (mock)\\n");
    return 0;
}}
"""
        else:
            code_template = f"""
// Generated Code for {target.value} - Iteration {iteration}
#include <stdio.h>
#include <stdlib.h>

void run_inference(float* input, float* output, int size) {{
    for(int i=0; i<size; i++) {{
        output[i] = input[i] * {0.5 if iteration < 2 else 1.0}f;
    }}
}}

int main() {{
    printf("Inference complete (mock)\\n");
    return 0;
}}
"""
        return code_template

import subprocess

class Compiler:
    def __init__(self):
        self.ndk_path = self._find_android_ndk()
    
    def _strip_markdown(self, code: str) -> str:
        """Remove markdown code block markers and explanatory text from generated code"""
        import re
        
        # Remove opening code fence (```c, ```cpp, ```C, etc.)
        code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        
        # Remove closing code fence
        code = re.sub(r'\n```\s*$', '', code)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any stray ``` markers in the middle
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Skip lines that are just ```
            if stripped == '```' or stripped.startswith('```') and len(stripped) < 10:
                continue
            lines.append(line)
        
        code = '\n'.join(lines)
        
        # CRITICAL: Remove AI explanatory text before/after code
        # Look for the first real C/C++ code marker (usually #include or /*)
        code_start_patterns = [r'^\s*#include', r'^\s*#define', r'^\s*/\*', r'^\s*//', r'^\s*typedef', r'^\s*struct', r'^\s*int\s+main', r'^\s*void\s+', r'^\s*float\s+']
        
        lines = code.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find where actual code starts (skip AI explanations like "Here is...")
        for i, line in enumerate(lines):
            if any(re.match(pattern, line) for pattern in code_start_patterns):
                start_idx = i
                break
        
        # Find where actual code ends (before explanations like "This code does...")
        # Heuristic: after the last closing brace at column 0, ignore remaining text
        last_brace_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '}' or lines[i].startswith('}'):
                last_brace_idx = i
                break
        
        if last_brace_idx > start_idx:
            # Check if there's substantial text after the last brace
            remaining_lines = lines[last_brace_idx + 1:]
            non_empty = [l for l in remaining_lines if l.strip() and not l.strip().startswith('//')]
            
            # If there are more than 2 non-empty lines after last brace, likely explanatory text
            if len(non_empty) > 2:
                # Check if these lines look like natural language (contain common words)
                explanation_keywords = ['this', 'the', 'code', 'implementation', 'performs', 'using', 'please', 'ensure', 'note']
                if any(any(keyword in line.lower() for keyword in explanation_keywords) for line in non_empty[:3]):
                    end_idx = last_brace_idx + 1
                    print(f"Stripped explanatory text after code (removed {len(non_empty)} lines)")
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def _sanitize_code(self, code: str, target: TargetType = None) -> str:
        """Remove problematic includes and C++ syntax that would prevent compilation"""
        import re
        
        # First strip markdown
        code = self._strip_markdown(code)
        
        # For OpenCL target, don't sanitize - allow OpenCL headers and C++ features
        if target == TargetType.OPENCL:
            return code
        
        original_code = code
        
        # AGGRESSIVELY remove raw string literals using regex - they span multiple lines
        # Match R"delimiter(...any content...)delimiter" where delimiter can be CL, EOF, etc.
        code = re.sub(r'R"([A-Z]*)\(.*?\)\1"', '""  /* raw string removed */', code, flags=re.DOTALL)
        
        if 'R"' in original_code:
            print(f"Sanitizer: Detected and removing raw string literals from generated code")
        
        # Also remove any lines that still have R" syntax
        lines = code.split('\n')
        sanitized = []
        skip_mode = False
        
        for line in lines:
            # If we see R" at the start of a line, enter skip mode
            if 'R"' in line:
                skip_mode = True
                sanitized.append(f"// {line.strip()}  // REMOVED: Raw string literal (C++ only)")
                continue
            
            # If in skip mode, keep skipping until we see the end
            if skip_mode:
                if '";' in line or ')CL";' in line or ')EOF";' in line:
                    skip_mode = False
                    sanitized.append(f"// {line.strip()}  // REMOVED: End of raw string")
                else:
                    sanitized.append(f"// {line.strip()}  // REMOVED: Raw string content")
                continue
            
            # Skip problematic includes
            if any(x in line for x in ['#include <CL/', '#include <cuda', '#include <cl_', 
                                        '#include <iostream>', '#include <vector>', '#include <string>']):
                sanitized.append(f"// {line.strip()}  // REMOVED: Unsupported include")
            else:
                sanitized.append(line)
        
        return '\n'.join(sanitized)
        
    def _find_android_ndk(self):
        """Find Android NDK path from environment or common locations"""
        ndk_path = os.environ.get("ANDROID_NDK")
        if ndk_path and os.path.isdir(ndk_path):
            return ndk_path
            
        # Try common locations
        home = os.path.expanduser("~")
        common_paths = [
            os.path.join(home, "Library/Android/sdk/ndk"),  # macOS
            os.path.join(home, "Android/Sdk/ndk"),  # Linux
        ]
        
        for base in common_paths:
            if os.path.isdir(base):
                # Find latest NDK version
                ndks = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
                if ndks:
                    return os.path.join(base, ndks[-1])
        return None
    
    def _get_android_clang(self, ndk_path, android_api="21", android_arch="aarch64"):
        """Find Android clang compiler in NDK (defaults to ARM64 for modern devices/GPUs)"""
        import platform
        
        # Detect host architecture
        system = platform.system().lower()
        machine = platform.machine()
        host_arch = f"{system}-{machine}"
        
        toolchain_dir = os.path.join(ndk_path, "toolchains/llvm/prebuilt", host_arch)
        if not os.path.isdir(toolchain_dir):
            # Try x86_64 variant
            toolchain_dir = os.path.join(ndk_path, "toolchains/llvm/prebuilt", f"{system}-x86_64")
        
        # Try specific architecture first (e.g., aarch64-linux-android21-clang)
        clang = os.path.join(toolchain_dir, "bin", f"{android_arch}-linux-android{android_api}-clang")
        if os.path.isfile(clang):
            print(f"✓ Found ARM64 NDK compiler: {os.path.basename(clang)}")
            return clang
        
        # Try with higher API levels if specified one doesn't exist
        for api in ["29", "28", "26", "24", "21", "19", "18"]:
            clang = os.path.join(toolchain_dir, "bin", f"{android_arch}-linux-android{api}-clang")
            if os.path.isfile(clang):
                print(f"✓ Found ARM64 NDK compiler: {os.path.basename(clang)}")
                return clang
        
        # Fallback to generic clang (will need -target flag)
        clang = os.path.join(toolchain_dir, "bin/clang")
        if os.path.isfile(clang):
            print(f"⚠️  Using generic clang (will specify -target)")
            return clang
        
        return None
    
    def compile(
        self,
        source_code: str,
        target: TargetType,
        compiler_cmd: str = "gcc",
        toolchain_config: Optional[Any] = None,
        kernel_code: Optional[str] = None,
        extra_include_paths: Optional[List[str]] = None,
    ):
        """
        Compiles source code with toolchain support for cross-compilation.
        Returns: (binary_path, kernel_path) tuple, or just binary_path for backward compatibility
        """
        # Determine file extension based on target
        is_opencl = target == TargetType.OPENCL
        
        # SANITIZE the code before compilation to remove C++ features (but skip for OpenCL)
        source_code = self._sanitize_code(source_code, target)
        src_ext = ".cpp" if is_opencl else ".c"
        
        # Save source to temp file
        src_path = f"/tmp/source_{uuid.uuid4().hex}{src_ext}"
        bin_path = f"/tmp/binary_{uuid.uuid4().hex}.bin"
        kernel_path = None
        
        with open(src_path, "w") as f:
            f.write(source_code)
        
        # Write kernel code to .cl file if provided
        if kernel_code and is_opencl:
            kernel_path = f"/tmp/kernel_{uuid.uuid4().hex}.cl"
            with open(kernel_path, "w") as f:
                f.write(kernel_code)
            print(f"Wrote OpenCL kernel to: {kernel_path}")
        
        compilation_errors = []
        
        extra_include_paths = extra_include_paths or []

        # Try custom toolchain if provided
        if toolchain_config and compiler_cmd != "mock":
            # For OpenCL: Ensure we're using a 64-bit ARM compiler (modern GPUs are 64-bit)
            if is_opencl and "armv7" in toolchain_config.compiler_path.lower():
                print(f"⚠️  Warning: Skipping 32-bit ARM compiler for OpenCL (GPUs require 64-bit)")
                print(f"   Compiler: {toolchain_config.compiler_path}")
                print(f"   Will try ARM64 NDK compiler instead...")
                compilation_errors.append("32-bit ARM compiler not suitable for OpenCL - need ARM64")
            else:
                try:
                    # Choose the right compiler: C++ for OpenCL, C for everything else
                    compiler_path = toolchain_config.compiler_path
                    if not is_opencl and compiler_path.endswith("clang++"):
                        # For non-OpenCL C code, use clang instead of clang++
                        compiler_path = compiler_path[:-2]  # Remove the "++"
                        print(f"Switching to C compiler: {compiler_path}")
                    
                    cmd = [compiler_path]
                    
                    # Add sysroot if specified
                    if toolchain_config.sysroot:
                        cmd.extend(["--sysroot", toolchain_config.sysroot])
                    
                    # Add include paths
                    for include_path in toolchain_config.include_paths:
                        cmd.extend(["-I", include_path])

                    # Add extra include paths (e.g., project root/includes)
                    for include_path in extra_include_paths:
                        cmd.extend(["-I", include_path])
                    
                    # Add OpenCL include path if this is an OpenCL target
                    if is_opencl:
                        from backend.toolchain_discovery import ToolchainDiscovery
                        discovery = ToolchainDiscovery()
                        sdks = discovery.discover_gpu_sdks()
                        opencl = sdks.get("opencl", {})
                        if opencl.get("headers_path"):
                            cmd.extend(["-I", opencl["headers_path"]])
                    
                    # Add library paths
                    for lib_path in toolchain_config.library_paths:
                        cmd.extend(["-L", lib_path])
                    
                    # Add compiler flags
                    cmd.extend(toolchain_config.compiler_flags)
                    
                    # Add standard flags (C++ for OpenCL, C for others)
                    std_flag = "-std=c++14" if is_opencl else "-std=c11"
                    cmd.extend([std_flag, "-o", bin_path, src_path])
                    
                    # Add linker flags
                    cmd.extend(toolchain_config.linker_flags)
                    
                    # For OpenCL: Static linking and undefined symbols
                    if is_opencl:
                        cmd.extend(["-static-libstdc++", "-static-libgcc"])
                        cmd.extend(["-Wl,--allow-shlib-undefined", "-Wl,--unresolved-symbols=ignore-all"])
                    
                    print(f"Compiling with toolchain: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Successfully compiled with custom toolchain: {bin_path}")
                        return (bin_path, kernel_path) if kernel_path else bin_path
                    else:
                        error_msg = f"Custom toolchain compilation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                        print(error_msg)
                        compilation_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Custom toolchain error: {e}"
                    print(error_msg)
                    compilation_errors.append(error_msg)
        
        # Try Android NDK cross-compilation if available
        if self.ndk_path and compiler_cmd != "mock":
            android_clang = self._get_android_clang(self.ndk_path)
            if android_clang:
                try:
                    print(f"Using Android NDK clang: {android_clang}")
                    cmd = [android_clang, "-target", "aarch64-linux-android21"]
                    
                    # Source file must come before output for proper linking
                    cmd.append(src_path)
                    
                    # OpenCL-specific compilation
                    if is_opencl:
                        cmd.extend(["-std=c++14"])  # Use C++ for OpenCL
                        
                        # Add OpenCL include paths
                        from backend.toolchain_discovery import ToolchainDiscovery
                        discovery = ToolchainDiscovery()
                        sdks = discovery.discover_gpu_sdks()
                        opencl = sdks.get("opencl", {})
                        if opencl.get("headers_path"):
                            cmd.extend(["-I", opencl["headers_path"]])
                            print(f"Using OpenCL headers from: {opencl['headers_path']}")

                        # Add extra include paths (e.g., project root/includes)
                        for include_path in extra_include_paths:
                            cmd.extend(["-I", include_path])
                        
                        # Statically link C++ runtime so we don't need libc++_shared.so on device
                        cmd.extend(["-static-libstdc++", "-static-libgcc"])
                        print("Note: Statically linking C++ runtime for self-contained binary")
                        
                        # CRITICAL: Don't link OpenCL library at compile time!
                        # Allow undefined symbols - they'll be resolved at runtime on device
                        # The device has /system/lib64/libOpenCL.so which will be dynamically linked
                        cmd.extend(["-Wl,--allow-shlib-undefined", "-Wl,--unresolved-symbols=ignore-all"])
                        print("Note: OpenCL symbols will be resolved at runtime by device's dynamic linker")
                    else:
                        cmd.extend(["-std=c11"])  # Use C11 for non-OpenCL
                        cmd.extend(["-static"])  # Statically link for non-OpenCL
                    
                    cmd.extend(["-o", bin_path])
                    
                    print(f"Compiling with NDK: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Successfully cross-compiled for Android: {bin_path}")
                        # Verify binary was created
                        if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
                            return (bin_path, kernel_path) if kernel_path else bin_path
                        else:
                            error_msg = "Binary file was not created or is empty"
                            print(error_msg)
                            compilation_errors.append(error_msg)
                    else:
                        error_msg = f"Android NDK compilation failed:\nCommand: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                        print(error_msg)
                        compilation_errors.append(error_msg)
                except FileNotFoundError as e:
                    error_msg = f"Android NDK compiler not found: {e}"
                    print(error_msg)
                    compilation_errors.append(error_msg)
        
        # Try regular compilation if not mock mode
        if compiler_cmd != "mock":
            try:
                # Determine compiler and flags based on target
                if is_opencl:
                    # Use g++ for OpenCL C++ code
                    cpp_compiler = "g++" if compiler_cmd == "gcc" else compiler_cmd
                    cmd = [cpp_compiler, "-std=c++14", "-o", bin_path, src_path]
                    
                    # Add OpenCL include and library paths
                    from backend.toolchain_discovery import ToolchainDiscovery
                    discovery = ToolchainDiscovery()
                    sdks = discovery.discover_gpu_sdks()
                    opencl = sdks.get("opencl", {})
                    
                    if opencl.get("headers_path"):
                        cmd.extend(["-I", opencl["headers_path"]])

                    # Add extra include paths (e.g., project root/includes)
                    for include_path in extra_include_paths:
                        cmd.extend(["-I", include_path])
                    
                    # Platform-specific OpenCL linking
                    import platform
                    if platform.system() == "Darwin":
                        cmd.extend(["-framework", "OpenCL"])
                    else:
                        cmd.append("-lOpenCL")
                else:
                    # Use gcc for C files (don't use -static on macOS as it causes linker issues)
                    cmd = [compiler_cmd, "-std=c11", "-o", bin_path, src_path]
                    for include_path in extra_include_paths:
                        cmd.extend(["-I", include_path])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return (bin_path, kernel_path) if kernel_path else bin_path
                else:
                    error_msg = f"Compilation with {compiler_cmd} failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    print(error_msg)
                    compilation_errors.append(error_msg)
            except FileNotFoundError as e:
                error_msg = f"Compiler {compiler_cmd} not found: {e}"
                print(error_msg)
                compilation_errors.append(error_msg)
        
        # ALWAYS raise error if compilation fails - NO MOCK MODE EVER
        all_errors = "\n\n".join(compilation_errors)
        error_summary = f"❌ ALL COMPILATION ATTEMPTS FAILED ({len(compilation_errors)} attempts)"
        raise Exception(f"{error_summary}\n\n{all_errors}")

class DeviceRunner:
    def deploy_and_run(self, binary_path, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Deploy and run on device. Supports SSH, ADB, or Local execution ONLY.
        NO MOCK MODE - Real execution required at all times.
        binary_path can be a string or a tuple (binary_path, kernel_path)
        """
        # Handle tuple return from compiler (for OpenCL)
        kernel_path = None
        if isinstance(binary_path, tuple):
            binary_path, kernel_path = binary_path
        
        # Check connection type
        if hasattr(config, 'connection_type'):
            if config.connection_type == 'ssh':
                return self._run_ssh(binary_path, input_data, config, expected_output, logs=logs, kernel_path=kernel_path)
            elif config.connection_type == 'adb':
                return self._run_adb(binary_path, input_data, config, expected_output, logs=logs, kernel_path=kernel_path)
            elif config.connection_type == 'local':
                return self._run_local(binary_path, input_data, expected_output, logs=logs, kernel_path=kernel_path)
        
        # Legacy support
        if config.use_adb:
            return self._run_adb(binary_path, input_data, config, expected_output, logs=logs, kernel_path=kernel_path)
        
        # Default to local execution
        return self._run_local(binary_path, input_data, expected_output, logs=logs, kernel_path=kernel_path)

    def _run_local(self, binary_path: str, input_data: np.ndarray, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None, kernel_path: Optional[str] = None) -> np.ndarray:
        """
        Run binary locally on host machine (macOS/Linux).
        """
        if logs is None:
            logs = []
        
        # Write input data to file
        input_file = binary_path + ".input"
        input_data.tofile(input_file)
        
        # Create symlink for input.bin in same directory as binary
        binary_dir = os.path.dirname(binary_path)
        if not binary_dir:
            binary_dir = "/tmp"
        local_input = os.path.join(binary_dir, "input.bin")
        local_output = os.path.join(binary_dir, "output.bin")
        local_kernel = os.path.join(binary_dir, "kernel.cl")
        
        # Copy input file
        shutil.copy(input_file, local_input)
        
        # Copy kernel file if provided (for OpenCL)
        if kernel_path and os.path.exists(kernel_path):
            # Be robust: some projects load "kernel.cl", others load "<basename>.cl".
            shutil.copy(kernel_path, local_kernel)
            try:
                local_kernel_basename = os.path.join(binary_dir, os.path.basename(kernel_path))
                if os.path.abspath(local_kernel_basename) != os.path.abspath(local_kernel):
                    shutil.copy(kernel_path, local_kernel_basename)
            except Exception:
                pass
            if logs is not None:
                logs.append({"stage": "Local Setup", "status": "Success", "details": f"Copied kernel to {local_kernel} (and basename if needed)"})
        
        try:
            if logs is not None:
                logs.append({"stage": "Local Execute", "status": "Running", "details": f"Executing {os.path.basename(binary_path)} locally..."})
            
            # Make sure binary is executable
            os.chmod(binary_path, 0o755)
            
            # Execute binary locally
            import time
            start_time = time.time()
            result = subprocess.run([binary_path], capture_output=True, text=True, timeout=30, cwd=binary_dir)
            execution_time = time.time() - start_time
            
            if logs is not None:
                if result.returncode == 0:
                    logs.append({"stage": "Local Execute", "status": "Success", "details": f"Execution complete in {execution_time:.3f}s"})
                    
                    # Add detailed output
                    if result.stdout or result.stderr:
                        device_output = ""
                        if result.stdout:
                            device_output += f"STDOUT:\n{result.stdout}"
                        if result.stderr:
                            device_output += f"\n\nSTDERR:\n{result.stderr}" if result.stdout else f"STDERR:\n{result.stderr}"
                        
                        logs.append({
                            "stage": "Device Output",
                            "status": "Info",
                            "details": device_output,
                            "execution_time": f"{execution_time:.3f}s"
                        })
                        print(f"\n{'='*60}\nLOCAL EXECUTION OUTPUT:\n{'='*60}\n{device_output}\n{'='*60}\n")
                else:
                    error_msg = f"Binary execution failed with exit code {result.returncode} (took {execution_time:.3f}s)\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    logs.append({"stage": "Local Execute", "status": "Failed", "details": error_msg})
                    raise RuntimeError(error_msg)
            
            # Read output file
            if os.path.exists(local_output) and os.path.getsize(local_output) > 0:
                output_array = np.fromfile(local_output, dtype=np.float32)
                
                # Validate size matches expectation
                if expected_output is not None:
                    expected_size = expected_output.size
                    if len(output_array) != expected_size:
                        error_msg = f"Size mismatch: Parsed {len(output_array)} values but expected {expected_size} values."
                        if logs is not None:
                            logs.append({"stage": "Output Parsing", "status": "Failed", "details": error_msg})
                        raise ValueError(error_msg)
                    
                    # Reshape to expected shape
                    output_array = output_array.reshape(expected_output.shape)
                
                if logs is not None:
                    logs.append({"stage": "Output Parsing", "status": "Success", 
                               "details": f"Parsed {len(output_array)} float32 values, shape: {output_array.shape}"})
                
                return output_array
            else:
                error_msg = "Output file not found or empty"
                if logs is not None:
                    logs.append({"stage": "Output Parsing", "status": "Failed", "details": error_msg})
                raise FileNotFoundError(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "Binary execution timed out (30s)"
            if logs is not None:
                logs.append({"stage": "Local Execute", "status": "Failed", "details": error_msg})
            raise RuntimeError(error_msg)
        finally:
            # Cleanup
            if os.path.exists(local_input):
                os.remove(local_input)
            if os.path.exists(local_output):
                os.remove(local_output)
            if kernel_path and os.path.exists(local_kernel):
                os.remove(local_kernel)
    
    def _run_mock(self, binary_path: str, input_data: np.ndarray, expected_output: Optional[np.ndarray] = None) -> np.ndarray:
        time.sleep(1) # Simulate
        
        # If we have expected output (from reference), use its shape/values for the mock
        if expected_output is not None:
            target_shape = expected_output.shape
            ref_vals = expected_output
        else:
            target_shape = input_data.shape
            ref_vals = input_data

        # Try to read as text to determine iteration (only works for mock binaries)
        try:
            with open(binary_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "Iteration 0" in content:
                return np.zeros(target_shape, dtype=np.float32)
            elif "Iteration 1" in content:
                # halfway
                return ref_vals * 0.5
            elif "Iteration 2" in content:
                # match
                return ref_vals * 1.0
        except (UnicodeDecodeError, Exception):
            # If it's a real binary, we can't read it as text
            # Return default mock output
            pass
        
        return ref_vals * 0.8

    def _run_ssh(self, binary_path: str, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None, kernel_path: Optional[str] = None) -> np.ndarray:
        """
        Deploy and run binary on remote device via SSH.
        """
        import paramiko
        
        if logs is None:
            logs = []
        
        # Write input data to file
        input_file = binary_path + ".input"
        input_data.tofile(input_file)
        
        remote_bin = f"{config.remote_work_dir}/{os.path.basename(binary_path)}"
        remote_input = f"{config.remote_work_dir}/input.bin"
        remote_out = f"{config.remote_work_dir}/output.bin"
        
        try:
            # Create SSH client
            if logs is not None:
                logs.append({"stage": "SSH Connect", "status": "Running", "details": f"Connecting to {config.ssh_host}:{config.ssh_port}..."})
            
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect with password or key
            connect_kwargs = {
                'hostname': config.ssh_host,
                'port': config.ssh_port,
                'username': config.ssh_user
            }
            
            if config.ssh_key_path:
                connect_kwargs['key_filename'] = os.path.expanduser(config.ssh_key_path)
            elif config.ssh_password:
                connect_kwargs['password'] = config.ssh_password
            
            ssh.connect(**connect_kwargs)
            
            if logs is not None:
                logs.append({"stage": "SSH Connect", "status": "Success", "details": "Connected to remote device"})
            
            # Create SFTP client for file transfer
            sftp = ssh.open_sftp()
            
            # Create remote work directory
            try:
                sftp.mkdir(config.remote_work_dir)
            except IOError:
                pass  # Directory might already exist
            
            # Upload binary
            if logs is not None:
                logs.append({"stage": "SSH Upload", "status": "Running", "details": f"Uploading binary to {remote_bin}..."})
            
            sftp.put(binary_path, remote_bin)
            sftp.chmod(remote_bin, 0o755)
            
            if logs is not None:
                logs.append({"stage": "SSH Upload", "status": "Success", "details": "Binary uploaded"})
            
            # Upload input data
            if logs is not None:
                logs.append({"stage": "SSH Upload Input", "status": "Running", "details": "Uploading input data..."})
            
            sftp.put(input_file, remote_input)
            
            if logs is not None:
                logs.append({"stage": "SSH Upload Input", "status": "Success", "details": "Input data uploaded"})
            
            # Execute binary
            if logs is not None:
                logs.append({"stage": "SSH Execute", "status": "Running", "details": f"Executing {os.path.basename(binary_path)}..."})
            
            import time
            start_time = time.time()
            stdin, stdout, stderr = ssh.exec_command(f"cd {config.remote_work_dir} && ./{os.path.basename(binary_path)}")
            exit_code = stdout.channel.recv_exit_status()
            execution_time = time.time() - start_time
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            if exit_code == 0:
                if logs is not None:
                    logs.append({"stage": "SSH Execute", "status": "Success", "details": f"Execution complete in {execution_time:.3f}s"})
                    
                    # Add detailed device output
                    if stdout_text or stderr_text:
                        device_output = ""
                        if stdout_text:
                            device_output += f"STDOUT:\n{stdout_text}"
                        if stderr_text:
                            device_output += f"\n\nSTDERR:\n{stderr_text}" if stdout_text else f"STDERR:\n{stderr_text}"
                        
                        logs.append({
                            "stage": "Device Output",
                            "status": "Info",
                            "details": device_output,
                            "execution_time": f"{execution_time:.3f}s"
                        })
                        print(f"\n{'='*60}\nDEVICE EXECUTION OUTPUT (SSH):\n{'='*60}\n{device_output}\n{'='*60}\n")
            else:
                error_msg = f"Execution failed with exit code {exit_code} (took {execution_time:.3f}s)\nSTDOUT: {stdout_text}\nSTDERR: {stderr_text}"
                if logs is not None:
                    logs.append({"stage": "SSH Execute", "status": "Failed", "details": error_msg})
                raise RuntimeError(error_msg)
            
            # Download output file
            if logs is not None:
                logs.append({"stage": "SSH Download", "status": "Running", "details": "Downloading output..."})
            
            local_out = binary_path + ".out"
            sftp.get(remote_out, local_out)
            
            if logs is not None:
                logs.append({"stage": "SSH Download", "status": "Success", "details": "Output downloaded"})
            
            # Parse output
            if os.path.exists(local_out) and os.path.getsize(local_out) > 0:
                output_array = np.fromfile(local_out, dtype=np.float32)
                
                # Validate and reshape
                if expected_output is not None:
                    expected_size = expected_output.size
                    if len(output_array) != expected_size:
                        error_msg = f"Size mismatch: Got {len(output_array)} values, expected {expected_size}"
                        if logs is not None:
                            logs.append({"stage": "Output Parsing", "status": "Failed", "details": error_msg})
                        raise ValueError(error_msg)
                    output_array = output_array.reshape(expected_output.shape)
                
                if logs is not None:
                    logs.append({"stage": "Output Parsing", "status": "Success", "details": f"Parsed {len(output_array)} values"})
                
                return output_array
            else:
                error_msg = "Output file not found or empty"
                if logs is not None:
                    logs.append({"stage": "Output Parsing", "status": "Failed", "details": error_msg})
                raise FileNotFoundError(error_msg)
            
        except Exception as e:
            error_msg = f"SSH deployment failed: {str(e)}"
            if logs is not None:
                logs.append({"stage": "SSH Error", "status": "Failed", "details": error_msg})
            # Re-raise so AI can fix the error
            raise
        finally:
            try:
                sftp.close()
                ssh.close()
            except:
                pass

    def _run_adb(self, binary_path: str, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None, kernel_path: Optional[str] = None) -> np.ndarray:
        """
        Deploy and run binary on Android device via ADB.
        Returns output from device execution.
        """
        import subprocess
        from backend.toolchain_discovery import ADBDiscovery
        
        # ALWAYS use ADBDiscovery to find ADB (consistent with compilation phase)
        ADB_PATH = config.adb_path if (hasattr(config, 'adb_path') and config.adb_path) else ADBDiscovery.find_adb()
        
        if not ADB_PATH:
            raise Exception("❌ ADB not found on system.\n\nPlease:\n1. Install Android SDK Platform Tools\n2. Restart the server\n3. Or manually set ADB path in Connection Settings")
        
        if logs is None:
            logs = []
        
        # Check if ADB device is connected
        try:
            check_cmd = [ADB_PATH, "devices"]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise Exception(f"ADB command failed: {result.stderr}")
            
            # Parse devices output
            lines = result.stdout.strip().split('\n')[1:]  # Skip "List of devices attached"
            devices = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == 'device':
                    devices.append(parts[0])
            
            if not devices:
                raise Exception("❌ No Android devices connected via ADB.\n\nPlease:\n1. Connect an Android device via USB\n2. Enable USB debugging on the device\n3. Run 'adb devices' to verify connection\n\nOr select 'Local' connection type to test on host machine instead.")
            
            print(f"✓ Found ADB devices: {devices}")
            
        except FileNotFoundError:
            raise Exception(f"❌ ADB not found at '{ADB_PATH}'.\n\nPlease:\n1. Install Android SDK Platform Tools\n2. Use 'Find ADB on Host' button in Connection Settings\n3. Or manually set the ADB path")
        except subprocess.TimeoutExpired:
            raise Exception(f"❌ ADB command timed out.\n\nCheck if ADB is working: {ADB_PATH}")
        
        # Write input data to file
        input_file = binary_path + ".input"
        input_data.tofile(input_file)
        
        remote_bin = f"{config.remote_work_dir}/{os.path.basename(binary_path)}"
        remote_input = f"{config.remote_work_dir}/input.bin"
        remote_out = f"{config.remote_work_dir}/output.bin"
        # Kernel naming: projects may open a specific filename like "simple_opencl.cl".
        # We will push the kernel twice: once as its original basename and once as "kernel.cl".
        remote_kernel = f"{config.remote_work_dir}/kernel.cl"
        remote_kernel_named = f"{config.remote_work_dir}/{os.path.basename(kernel_path)}" if kernel_path else None
        
        adb_cmd = [ADB_PATH]
        if config.adb_device_id:
            adb_cmd.extend(["-s", config.adb_device_id])
            
        try:
            # Step 1: Push binary to device
            if logs is not None:
                logs.append({"stage": "ADB Push", "status": "Running", "details": f"Copying {os.path.basename(binary_path)} to device..."})
            
            push_cmd = adb_cmd + ["push", binary_path, remote_bin]
            result = subprocess.run(push_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                logs.append({"stage": "ADB Push", "status": "Success", "details": f"Copied to {remote_bin}"})
            
            # Step 1.5: Diagnostic - check binary dependencies (for OpenCL debugging)
            if kernel_path:  # Only for OpenCL targets
                try:
                    diag_cmd = adb_cmd + ["shell", f"readelf -d {remote_bin} | grep NEEDED || file {remote_bin}"]
                    diag_result = subprocess.run(diag_cmd, capture_output=True, text=True, timeout=10)
                    if diag_result.stdout.strip():
                        print(f"📋 Binary dependencies:\n{diag_result.stdout.strip()}")
                except:
                    pass  # Diagnostic failure is non-fatal
            
            # Step 2: Push input data to device
            if logs is not None:
                logs.append({"stage": "ADB Push Input", "status": "Running", "details": f"Copying input data ({input_data.size} values) to device..."})
            
            push_input_cmd = adb_cmd + ["push", input_file, remote_input]
            subprocess.run(push_input_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                logs.append({"stage": "ADB Push Input", "status": "Success", "details": f"Input data copied to {remote_input}"})
            
            # Step 2.5: Push kernel file if provided (for OpenCL)
            if kernel_path and os.path.exists(kernel_path):
                if logs is not None:
                    logs.append({"stage": "ADB Push Kernel", "status": "Running", "details": f"Copying kernel to device..."})
                
                # Push under original basename first (so fopen(\"simple_opencl.cl\") works)
                if remote_kernel_named:
                    push_kernel_cmd = adb_cmd + ["push", kernel_path, remote_kernel_named]
                    subprocess.run(push_kernel_cmd, capture_output=True, text=True, check=True)
                # Also push/copy to kernel.cl for code that expects that name
                if not remote_kernel_named or remote_kernel_named != remote_kernel:
                    push_kernel_cmd = adb_cmd + ["push", kernel_path, remote_kernel]
                    subprocess.run(push_kernel_cmd, capture_output=True, text=True, check=True)
                
                if logs is not None:
                    logs.append({"stage": "ADB Push Kernel", "status": "Success", "details": f"Kernel copied to {remote_kernel_named or remote_kernel} (and kernel.cl fallback)"})
            
            # Step 2.6: Push OpenCL library if we have one (for OpenCL targets)
            # This is crucial because SELinux may prevent access to /vendor/lib64
            opencl_lib_pushed = False
            if kernel_path:  # Only for OpenCL targets
                # Get device ID - use configured one or detect from connected devices
                device_id = config.adb_device_id
                if not device_id and devices:
                    device_id = devices[0]  # Use first connected device
                    print(f"Using detected device ID: {device_id}")
                
                if device_id:
                    # Try to find the pulled OpenCL library for this device
                    device_lib_dir = os.path.join(os.path.dirname(__file__), "opencl_sdk", "libs", device_id)
                    local_opencl_lib = os.path.join(device_lib_dir, "libOpenCL.so")
                else:
                    local_opencl_lib = None
                    print("⚠️  No device ID found, will try system OpenCL paths")
                
                if local_opencl_lib and os.path.exists(local_opencl_lib):
                    if logs is not None:
                        logs.append({"stage": "ADB Push OpenCL", "status": "Running", "details": f"Copying libOpenCL.so to device..."})
                    
                    remote_opencl_lib = f"{config.remote_work_dir}/libOpenCL.so"
                    push_opencl_cmd = adb_cmd + ["push", local_opencl_lib, remote_opencl_lib]
                    result = subprocess.run(push_opencl_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        opencl_lib_pushed = True
                        if logs is not None:
                            logs.append({"stage": "ADB Push OpenCL", "status": "Success", "details": f"OpenCL library copied to {remote_opencl_lib}"})
                        print(f"✓ Pushed OpenCL library to device for local linking")
                        
                        # Verify the library on device
                        verify_cmd = adb_cmd + ["shell", f"file {remote_opencl_lib} || ls -lh {remote_opencl_lib}"]
                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
                        print(f"Library verification: {verify_result.stdout.strip()}")
                    else:
                        print(f"⚠️  Failed to push OpenCL library: {result.stderr}")
            
            # Step 3: Set executable permissions
            if logs is not None:
                logs.append({"stage": "ADB Chmod", "status": "Running", "details": "Setting executable permissions..."})
            
            chmod_cmd = adb_cmd + ["shell", "chmod", "+x", remote_bin]
            subprocess.run(chmod_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                logs.append({"stage": "ADB Chmod", "status": "Success", "details": "Permissions set"})
            
            # Step 3.5: Create and push a wrapper script that sets LD_LIBRARY_PATH
            # CRITICAL: Use system paths FIRST (SELinux often blocks loading from /data/local/tmp)
            # The copied library is just a backup for analysis
            lib_path_value = "/vendor/lib64:/system/lib64:/vendor/lib:/system/lib"
            if opencl_lib_pushed:
                # Add local path as LAST resort (often blocked by SELinux)
                lib_path_value += f":{config.remote_work_dir}"
                print(f"Using system OpenCL libraries (with local backup at {config.remote_work_dir})")
            else:
                print("Using system OpenCL library paths only")
            
            wrapper_script = f"""#!/system/bin/sh
cd {config.remote_work_dir}
export LD_LIBRARY_PATH={lib_path_value}

# AGGRESSIVE DIAGNOSTICS
echo "╔═══════════════════════════════════════╗"
echo "║  OpenCL Runtime Diagnostics           ║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "1. LD_LIBRARY_PATH:"
echo "   $LD_LIBRARY_PATH"
echo ""
echo "2. OpenCL library search:"
for lib_path in /vendor/lib64 /system/lib64 /vendor/lib /system/lib {config.remote_work_dir}; do
    if [ -f "$lib_path/libOpenCL.so" ]; then
        echo "   ✓ Found: $lib_path/libOpenCL.so"
        ls -lh "$lib_path/libOpenCL.so"
        file "$lib_path/libOpenCL.so" 2>&1 | head -1
    fi
done
echo ""
echo "3. Binary info:"
file ./{os.path.basename(binary_path)}
echo ""
echo "4. Binary dependencies:"
readelf -d ./{os.path.basename(binary_path)} | grep NEEDED || echo "   (no deps found)"
echo ""
echo "5. Executing with OpenCL..."
echo "═══════════════════════════════════════"
echo ""

# Use LD_PRELOAD by default (most reliable for OpenCL on Android)
if [ -f "/vendor/lib64/libOpenCL.so" ]; then
    export LD_PRELOAD="/vendor/lib64/libOpenCL.so"
    echo "✓ Using LD_PRELOAD=/vendor/lib64/libOpenCL.so"
elif [ -f "/system/lib64/libOpenCL.so" ]; then
    export LD_PRELOAD="/system/lib64/libOpenCL.so"
    echo "✓ Using LD_PRELOAD=/system/lib64/libOpenCL.so"
elif [ -f "{config.remote_work_dir}/libOpenCL.so" ]; then
    export LD_PRELOAD="{config.remote_work_dir}/libOpenCL.so"
    echo "✓ Using LD_PRELOAD={config.remote_work_dir}/libOpenCL.so"
fi

echo ""
./{os.path.basename(binary_path)}
EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════"
echo "Final exit code: $EXIT_CODE"
exit $EXIT_CODE
"""
            local_wrapper = f"/tmp/run_wrapper_{uuid.uuid4().hex}.sh"
            remote_wrapper = f"{config.remote_work_dir}/run_wrapper.sh"
            
            with open(local_wrapper, "w") as f:
                f.write(wrapper_script)
            
            # Push wrapper script
            push_wrapper_cmd = adb_cmd + ["push", local_wrapper, remote_wrapper]
            subprocess.run(push_wrapper_cmd, capture_output=True, text=True, check=True)
            
            # Make wrapper executable
            chmod_wrapper_cmd = adb_cmd + ["shell", "chmod", "+x", remote_wrapper]
            subprocess.run(chmod_wrapper_cmd, capture_output=True, text=True, check=True)
            
            # Clean up local wrapper
            try:
                os.remove(local_wrapper)
            except:
                pass
            
            # Step 4: Execute binary on device via wrapper script
            if logs is not None:
                logs.append({"stage": "ADB Execute", "status": "Running", "details": f"Running {os.path.basename(binary_path)} on device..."})
            
            import time
            start_time = time.time()
            exec_cmd = adb_cmd + ["shell", remote_wrapper]
            # IMPORTANT: do not use check=True here. If the wrapper exits non-zero,
            # we still need stdout/stderr for diagnosis and for the agent to iterate.
            exec_result = subprocess.run(exec_cmd, capture_output=True, text=True, check=False)
            execution_time = time.time() - start_time
            
            # Capture full device output for detailed logging
            device_stdout = exec_result.stdout if exec_result.stdout else ""
            device_stderr = exec_result.stderr if exec_result.stderr else ""
            
            if logs is not None:
                # Add execution summary (success/failure)
                status = "Success" if exec_result.returncode == 0 else "Failed"
                logs.append(
                    {
                        "stage": "ADB Execute",
                        "status": status,
                        "details": f"Execution finished in {execution_time:.3f}s | Exit code: {exec_result.returncode}",
                    }
                )
                
                # Add detailed device output as a separate log entry
                if device_stdout or device_stderr:
                    device_output = ""
                    if device_stdout:
                        device_output += f"STDOUT:\n{device_stdout}"
                    if device_stderr:
                        device_output += f"\n\nSTDERR:\n{device_stderr}" if device_stdout else f"STDERR:\n{device_stderr}"
                    
                    logs.append({
                        "stage": "Device Output", 
                        "status": "Info", 
                        "details": device_output,
                        "execution_time": f"{execution_time:.3f}s"
                    })
                    print(f"\n{'='*60}\nDEVICE EXECUTION OUTPUT:\n{'='*60}\n{device_output}\n{'='*60}\n")

            # If wrapper failed, raise with full device output (so the agent can fix root cause).
            if exec_result.returncode != 0:
                device_output = ""
                if device_stdout:
                    device_output += f"STDOUT:\n{device_stdout}"
                if device_stderr:
                    device_output += f"\n\nSTDERR:\n{device_stderr}" if device_stdout else f"STDERR:\n{device_stderr}"
                raise RuntimeError(
                    "ADB wrapper execution failed.\n"
                    f"Exit code: {exec_result.returncode}\n\n"
                    f"{device_output}".strip()
                )
            
            # Step 5: Pull output file from device
            if logs is not None:
                logs.append({"stage": "ADB Pull", "status": "Running", "details": f"Retrieving output from {remote_out}..."})
            
            local_out = binary_path + ".out"
            pull_cmd = adb_cmd + ["pull", remote_out, local_out]
            
            try:
                subprocess.run(pull_cmd, capture_output=True, text=True, check=True)
                
                if logs is not None:
                    logs.append({"stage": "ADB Pull", "status": "Success", "details": f"Output retrieved to {local_out}"})
                
                # Parse the output file (float32 binary format)
                try:
                    if os.path.exists(local_out) and os.path.getsize(local_out) > 0:
                        output_array = np.fromfile(local_out, dtype=np.float32)
                        num_parsed = len(output_array)
                        
                        # Validate size matches expectation
                        if expected_output is not None:
                            expected_size = expected_output.size
                            if num_parsed != expected_size:
                                error_msg = f"Size mismatch: Parsed {num_parsed} values but expected {expected_size} values. The generated C code likely only wrote {num_parsed} value(s) instead of the full array."
                                if logs is not None:
                                    logs.append({"stage": "Output Parsing", "status": "Failed", 
                                               "details": error_msg})
                                raise ValueError(error_msg)
                            
                            # Reshape to expected shape
                            try:
                                output_array = output_array.reshape(expected_output.shape)
                            except ValueError as e:
                                # Shape mismatch
                                if logs is not None:
                                    logs.append({"stage": "Output Parsing", "status": "Warning", 
                                               "details": f"Could not reshape output from {output_array.shape} to {expected_output.shape}: {str(e)}"})
                        
                        if logs is not None:
                            logs.append({"stage": "Output Parsing", "status": "Success", 
                                       "details": f"Parsed {num_parsed} float32 values, shape: {output_array.shape}"})
                        
                        return output_array
                    else:
                        if logs is not None:
                            logs.append({"stage": "Output Parsing", "status": "Warning", 
                                       "details": "Output file is empty or doesn't exist"})
                        raise FileNotFoundError("Empty or missing output file")
                        
                except Exception as e:
                    if logs is not None:
                        logs.append({"stage": "Output Parsing", "status": "Failed", 
                                   "details": f"Failed to parse output: {str(e)}"})
                    # Fall through to mock data
                    
            except subprocess.CalledProcessError:
                if logs is not None:
                    logs.append({"stage": "ADB Pull", "status": "Warning", "details": "No output file found on device"})
            
            # If we reached here, output parsing failed - raise exception for AI to fix
            raise FileNotFoundError("Binary executed but produced no valid output file. The code may have crashed or failed to write output.bin.")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ADB command failed: {e.stderr if e.stderr else str(e)}"
            if logs is not None:
                logs.append({"stage": "ADB Error", "status": "Failed", "details": error_msg})
            print(f"ADB Error: {error_msg}")
            # Re-raise so AI can fix the execution error
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logs is not None:
                logs.append({"stage": "ADB Error", "status": "Failed", "details": error_msg})
            print(f"ADB Error: {error_msg}")
            # Re-raise so AI can fix the error
            raise

class PortingEngine:
    def __init__(self):
        self.comp_engine = ComparisonEngine()
        self.generator = CodeGenerator()
        self.compiler = Compiler()
        self.runner = DeviceRunner()
        
    def run_porting_loop(self, 
                         source_model_path: str, 
                         target: TargetType, 
                         config: DeviceConfig,
                         max_iterations: int = 3,
                         input_shape: List[int] = [1, 5],
                         callback: Optional[Callable] = None,
                         job_id: Optional[str] = None,
                         debug_instructions: str = "") -> List[Dict[str, Any]]:
        
        logs = []
        
        def add_log(log_entry):
            add_log(log_entry)
            if callback:
                try:
                    asyncio.create_task(callback(log_entry))
                except RuntimeError:
                    # If no event loop is running, skip WebSocket broadcast
                    pass
        
        # Save artifacts helper
        def save_artifact(filename, content):
            if job_id:
                from backend.jobs.job_manager import JobManager
                jm = JobManager()
                if isinstance(content, str):
                    jm.save_artifact(job_id, filename, content.encode('utf-8'))
                elif isinstance(content, np.ndarray):
                    path = jm.get_job_dir(job_id) / filename
                    np.save(path, content)
                else:
                    jm.save_artifact(job_id, filename, content)
        
        # 1. Run Reference Model on HOST
        add_log({"stage": "Reference Model (HOST)", "status": "Loading", "details": f"Loading PyTorch model from {source_model_path}..."})
        # We need to load the model to get expected output
        # Use existing ComparisonEngine helper, but we need raw input/output
        model, _ = self.comp_engine.load_model_from_path(source_model_path)
        model.eval()
        add_log({"stage": "Reference Model (HOST)", "status": "Loaded", "details": "Model loaded successfully"})
        
        # Generate random input based on provided shape
        add_log({"stage": "Reference Model (HOST)", "status": "Running", "details": f"Executing PyTorch model on host machine with input shape {input_shape}..."})
        input_data = np.random.randn(*input_shape).astype(np.float32)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            ref_output_tensor = model(torch.from_numpy(input_data))
            ref_output = ref_output_tensor.numpy()
        
        # Save artifacts
        save_artifact("input.npy", input_data)
        save_artifact("host_output.npy", ref_output)
            
        add_log({"stage": "Reference Model (HOST)", "status": "Complete", "details": f"Execution finished | Output Shape: {ref_output.shape} | Device: {device} | Values: {ref_output.flatten()[:5].tolist()}"})
        
        # 2. Porting Loop
        best_diff = float('inf')
        compilation_history: List[str] = []
        
        for i in range(max_iterations):
            add_log({"stage": f"Iteration {i}", "status": "Generating Code", "details": ""})
            
            # Generate
            # We pass the FILE CONTENT as source logic
            with open(source_model_path, 'r') as f:
                src_content = f.read()
            code_result = self.generator.generate(src_content, target, i, debug_instructions=debug_instructions, device_config=config)
            
            # Handle dict return from generator
            if isinstance(code_result, dict):
                host_code = code_result.get("host_code", "")
                kernel_code = code_result.get("kernel_code")
            else:
                # Backward compatibility
                host_code = code_result
                kernel_code = None
            
            add_log({"stage": f"Iteration {i}", "status": "Code Generated", "source_preview": host_code[:500]})
            
            # Save generated code
            file_ext = ".cpp" if target == TargetType.OPENCL else ".c"
            save_artifact(f"generated_code_iter_{i}{file_ext}", host_code)
            if kernel_code:
                save_artifact(f"kernel_iter_{i}.cl", kernel_code)
            
            # Compile
            add_log({"stage": f"Iteration {i}", "status": "Compiling", "details": f"Target: {target.value}"})
            try:
                toolchain = config.toolchain if hasattr(config, 'toolchain') else None
                binary = self.compiler.compile(host_code, target, config.compiler_cmd, toolchain_config=toolchain, kernel_code=kernel_code)
            except Exception as e:
                err = str(e)
                add_log({"stage": f"Iteration {i}", "status": "Compilation Failed", "details": err})
                compilation_history.append(err[:120])

                if len(compilation_history) >= 3:
                    last_three = compilation_history[-3:]
                    if all(sig == last_three[0] for sig in last_three):
                        add_log(
                            {
                                "stage": "Result",
                                "status": "Failed",
                                "details": f"⚠️ STUCK IN LOOP: Same compilation error repeated 3 times.\n\nError: {last_three[0]}\n\nThis is likely an environment/dependency issue (headers/toolchain), not a code issue.",
                            }
                        )
                        break
                continue
                
            # Run on TARGET device
            add_log({"stage": f"Iteration {i} (TARGET)", "status": "Deploying & Running", "details": f"Executing compiled binary on target device..."})
            try:
                actual_output = self.runner.deploy_and_run(binary, input_data, config, expected_output=ref_output, logs=logs)
            except Exception as e:
                add_log({"stage": f"Iteration {i} (TARGET)", "status": "Execution Failed", "details": str(e)})
                continue
            
            # Save target output
            save_artifact(f"target_output_iter_{i}.npy", actual_output)
                
            # Compare - validate shapes first
            if ref_output.shape != actual_output.shape:
                add_log({"stage": f"Iteration {i}", "status": "Shape Mismatch", "details": f"HOST shape: {ref_output.shape} vs TARGET shape: {actual_output.shape}"})
                add_log({"stage": "Result", "status": "Failed", "details": f"Shape mismatch: Reference outputs {ref_output.shape} but generated code outputs {actual_output.shape}"})
                break
                
            diff = np.linalg.norm(ref_output - actual_output)
            add_log({"stage": f"Iteration {i}", "status": "Verified", "details": f"L2 Diff: {diff:.4f} | HOST shape: {ref_output.shape} | TARGET shape: {actual_output.shape}"})
            
            best_diff = min(best_diff, diff)
            
            if diff < 1e-4:
                add_log({"stage": "Result", "status": "Success", "details": "Porting Converged!"})
                break
                
        if best_diff >= 1e-4:
             add_log({"stage": "Result", "status": "Failed", "details": "Could not converge within max iterations."})
             
        return logs

    def verify_manual_code(self, 
                           manual_source_path: str,
                           reference_model_path: str,
                           target: TargetType,
                           config: DeviceConfig,
                           input_shape: List[int] = [1, 5],
                           max_iterations: int = 3,
                           callback: Optional[Callable] = None,
                           job_id: Optional[str] = None,
                           debug_instructions: str = "") -> List[Dict[str, Any]]:
        """
        Verifies manually uploaded code - cross-compiles and deploys to target hardware.
        If reference model is provided and there's a mismatch, will iterate and try to fix the code.
        """
        logs = []
        
        def add_log(log_entry):
            logs.append(log_entry)
            print(f"[LOG] {log_entry.get('stage', 'Unknown')}: {log_entry.get('status', 'Unknown')} - {log_entry.get('details', '')[:100]}")
            if callback:
                try:
                    # Get the running event loop and schedule the coroutine
                    import concurrent.futures
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(callback(log_entry), loop)
                except RuntimeError:
                    # If no event loop is running, try to get any event loop
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(callback(log_entry), loop)
                    except Exception as e:
                        print(f"Warning: Failed to send log callback: {e}")
        
        def save_artifact(filename, content):
            if job_id:
                from backend.jobs.job_manager import JobManager
                jm = JobManager()
                if isinstance(content, str):
                    jm.save_artifact(job_id, filename, content.encode('utf-8'))
                elif isinstance(content, np.ndarray):
                    path = jm.get_job_dir(job_id) / filename
                    np.save(path, content)
                else:
                    jm.save_artifact(job_id, filename, content)
        
        add_log({"stage": "Initialization", "status": "Starting Manual Verification", "details": f"Auto-fix enabled: {max_iterations} iterations max, Target: {target.value}"})
        
        # 1. Load initial source code and check if it's part of a project
        add_log({"stage": "Source Code", "status": "Loading", "details": f"Reading manual source code..."})
        project_context = None
        project_root = None
        manifest_path = None
        try:
            with open(manual_source_path, 'r') as f:
                src_content = f.read()
            
            # Check if this file is part of a project (walk up for .project_manifest.json)
            def _find_manifest_dir(start_dir: str) -> Optional[str]:
                cur = start_dir
                while True:
                    cand = os.path.join(cur, ".project_manifest.json")
                    if os.path.exists(cand):
                        return cur
                    parent = os.path.dirname(cur)
                    if parent == cur:
                        return None
                    cur = parent

            source_dir = os.path.dirname(manual_source_path)
            manifest_dir = _find_manifest_dir(source_dir)
            if manifest_dir:
                project_root = manifest_dir
                manifest_path = os.path.join(project_root, ".project_manifest.json")

                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                folder_name = manifest.get("folder_name", "project")
                upload_root = os.path.dirname(project_root)

                def _strip_folder_prefix(p: str) -> str:
                    prefix = folder_name + "/"
                    return p[len(prefix):] if isinstance(p, str) and p.startswith(prefix) else p

                project_files: Dict[str, str] = {}
                for p in manifest.get("files", []):
                    rel_inside = _strip_folder_prefix(p)
                    full_path = os.path.join(project_root, rel_inside)
                    # Fallback for legacy manifests storing upload_root-relative paths
                    if not os.path.exists(full_path):
                        full_path = os.path.join(upload_root, p)
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, "r") as pf:
                                project_files[rel_inside] = pf.read()
                        except Exception:
                            pass

                project_context = {
                    "folder_name": folder_name,
                    "files": project_files,
                    "main_file": _strip_folder_prefix(manifest.get("main_file", "") or ""),
                }

                add_log(
                    {
                        "stage": "Source Code",
                        "status": "Loaded",
                        "details": f"📁 Project root: {project_root} | Files loaded: {len(project_files)}",
                        "source_preview": src_content[:500] + "..." if len(src_content) > 500 else src_content,
                    }
                )
            else:
                add_log(
                    {
                        "stage": "Source Code",
                        "status": "Loaded",
                        "source_preview": src_content[:500] + "..." if len(src_content) > 500 else src_content,
                    }
                )
        except Exception as e:
            add_log({"stage": "Source Code", "status": "Failed", "details": str(e)})
            return logs

        # 2. Load reference model first (if provided) to get expected output shape
        ref_output = None
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        if reference_model_path and os.path.exists(reference_model_path):
            add_log({"stage": "Reference Model (HOST)", "status": "Loading", "details": "Loading PyTorch reference model..."})
            try:
                model, _ = self.comp_engine.load_model_from_path(reference_model_path)
                model.eval()
                add_log({"stage": "Reference Model (HOST)", "status": "Loaded", "details": "Model loaded successfully"})
                
                add_log({"stage": "Reference Model (HOST)", "status": "Running", "details": "Executing PyTorch model on host machine..."})
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.no_grad():
                    ref_output_tensor = model(torch.from_numpy(input_data))
                    ref_output = ref_output_tensor.numpy()
                
                add_log({"stage": "Reference Model (HOST)", "status": "Complete", "details": f"Execution finished | Output Shape: {ref_output.shape} | Device: {device} | Values: {ref_output.flatten()[:5].tolist()}"})
            except Exception as e:
                add_log({"stage": "Reference Model (HOST)", "status": "Failed", "details": f"Could not load reference: {str(e)}"})
                ref_output = None
        
        # 3. Iterative compile-test-fix loop with smart error tracking
        best_diff = float('inf')
        current_code = src_content
        current_kernel = None  # For OpenCL targets
        error_history = []  # Track normalized errors to detect infinite loops
        compilation_history = []  # Track normalized compile error signatures too

        entry_relpath = None
        extra_includes: List[str] = []
        project_profile = None
        build_plan = None
        nnport_run_id = None
        if project_root and project_context:
            # Analyze project to identify the best entrypoint (do not blindly trust manifest main_file).
            try:
                project_profile = analyze_project(project_root)
                if project_profile.selected_entrypoint:
                    entry_relpath = project_profile.selected_entrypoint
                    add_log(
                        {
                            "stage": "Project Analysis",
                            "status": "Complete",
                            "details": f"Selected entrypoint: {entry_relpath} (candidates: {len(project_profile.main_candidates)}) | CMake: {project_profile.has_cmake}",
                        }
                    )
                else:
                    entry_relpath = project_context.get("main_file") or ""
                    add_log(
                        {
                            "stage": "Project Analysis",
                            "status": "Warning",
                            "details": f"No main() found; falling back to manifest main_file: {entry_relpath or '(none)'}",
                        }
                    )
            except Exception as e:
                entry_relpath = project_context.get("main_file") or ""
                add_log(
                    {
                        "stage": "Project Analysis",
                        "status": "Warning",
                        "details": f"Project analysis failed ({e}); falling back to manifest main_file: {entry_relpath or '(none)'}",
                    }
                )
            if not entry_relpath:
                try:
                    entry_relpath = os.path.relpath(manual_source_path, project_root)
                except Exception:
                    entry_relpath = ""

            # Include project root, plus any header directories we can infer.
            extra_includes = [project_root]
            try:
                header_dirs = set()
                for relp in (project_context.get("files") or {}).keys():
                    if relp.endswith((".h", ".hpp", ".hh")):
                        header_dirs.add(os.path.join(project_root, os.path.dirname(relp)))
                extra_includes.extend(sorted(header_dirs))
            except Exception:
                pass

            # Create a stable run id (used for `.nnport/runs/<run_id>/...`).
            # Prefer job_id so users can find artifacts deterministically.
            nnport_run_id = job_id or "manual"

            # Persist profile for UI/diagnostics.
            try:
                if project_profile:
                    ws = get_nnport_workspace(project_root, create=True)
                    run_dir = os.path.join(ws.runs_root, nnport_run_id)
                    profile_path = write_profile(project_profile, run_dir)
                    try:
                        save_artifact("project_profile.json", json.dumps(project_profile.to_dict(), indent=2))
                    except Exception:
                        pass
                    add_log({"stage": "Project Analysis", "status": "Saved", "details": f"Profile saved: {profile_path}"})
            except Exception:
                pass
        else:
            # Non-project single-file mode: at minimum, include the directory of the provided source file
            # so local headers like "opencl_context.h" can be resolved.
            try:
                extra_includes = [os.path.dirname(manual_source_path)]
            except Exception:
                extra_includes = []
        
        for iteration in range(max_iterations):
            # Check if we're stuck in a loop (same error 3+ times)
            if len(error_history) >= 3:
                last_three = error_history[-3:]
                if all(e == last_three[0] for e in last_three):
                    add_log({"stage": "Result", "status": "Failed", "details": f"⚠️ STUCK IN LOOP: Same error repeated 3 times.\n\nError: {last_three[0]}\n\nThis is likely a system/infrastructure issue, not a code issue. The AI cannot fix this.\n\nSuggestions:\n1. Check if libOpenCL.so is correct architecture (32-bit vs 64-bit)\n2. Try running locally first (connection_type: local)\n3. Manually verify OpenCL works on device: adb shell 'ls -la /vendor/lib64/libOpenCL.so'\n4. Check SELinux policies"})
                    return logs
            add_log({"stage": f"Iteration {iteration}", "status": "Starting", "details": f"Testing code version {iteration}"})

            # Refresh project profile each iteration (files may have changed due to applied ops).
            if project_root and project_context:
                try:
                    project_profile = analyze_project(project_root)
                    if project_profile and project_profile.selected_entrypoint:
                        entry_relpath = project_profile.selected_entrypoint
                except Exception:
                    pass
            
            kernel_path_for_run = None
            # If we have a project root, the project folder is the source of truth.
            if project_root and entry_relpath:
                entry_abs = os.path.join(project_root, entry_relpath)
                try:
                    with open(entry_abs, "r") as f:
                        host_code = f.read()
                except Exception as e:
                    add_log({"stage": f"Iteration {iteration}", "status": "Failed", "details": f"Could not read entry file {entry_relpath}: {e}"})
                    return logs
                if target == TargetType.OPENCL:
                    cand = os.path.join(project_root, "kernel.cl")
                    if os.path.exists(cand):
                        kernel_path_for_run = cand
            else:
                # Handle dict format (from legacy AI fixes for OpenCL)
                if isinstance(current_code, dict):
                    host_code = current_code.get("host_code", "")
                    current_kernel = current_code.get("kernel_code")
                else:
                    host_code = current_code
            
            # Compile
            add_log({"stage": f"Iteration {iteration}", "status": "Compiling", "details": f"Target: {target.value}"})
            try:
                if project_root and project_profile:
                    # Build via NNPort-generated CMake harness under `.nnport/` (preferred policy).
                    opencl_include_root = None
                    if target == TargetType.OPENCL:
                        try:
                            from backend.toolchain_discovery import ToolchainDiscovery

                            sdks = ToolchainDiscovery().discover_gpu_sdks()
                            opencl_include_root = (sdks.get("opencl") or {}).get("headers_path")
                        except Exception:
                            opencl_include_root = None

                    build_plan = generate_cmake_harness(
                        project_profile,
                        ndk_path=self.compiler.ndk_path if getattr(self.compiler, "ndk_path", None) else None,
                        opencl_include_root=opencl_include_root,
                        run_id=nnport_run_id,
                    )
                    # Persist build plan for UI/diagnostics.
                    try:
                        if nnport_run_id:
                            ws = get_nnport_workspace(project_root, create=True)
                            run_dir = os.path.join(ws.runs_root, nnport_run_id)
                            os.makedirs(run_dir, exist_ok=True)
                            with open(os.path.join(run_dir, "build_plan.json"), "w", encoding="utf-8") as f:
                                json.dump(build_plan.to_dict(), f, indent=2)
                            try:
                                save_artifact("build_plan.json", json.dumps(build_plan.to_dict(), indent=2))
                            except Exception:
                                pass
                            # Also write a visible pointer file so users notice artifacts without needing `ls -a`.
                            try:
                                with open(os.path.join(project_root, "NNPORT_LAST_RUN.json"), "w", encoding="utf-8") as f2:
                                    json.dump(
                                        {
                                            "run_id": nnport_run_id,
                                            "run_dir": run_dir,
                                            "harness_dir": build_plan.harness_dir,
                                            "build_dir": build_plan.build_dir,
                                            "target_exe": build_plan.target_exe,
                                            "entrypoint": build_plan.entrypoint,
                                        },
                                        f2,
                                        indent=2,
                                    )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    add_log(
                        {
                            "stage": "Build Plan",
                            "status": "Ready",
                            "details": f"Mode: {build_plan.mode} | Entrypoint: {build_plan.entrypoint} | Sources: {len(build_plan.sources)} | Build dir: {build_plan.build_dir}",
                        }
                    )

                    exe_path = build_cmake_harness(build_plan)
                    if not elf_has_main(exe_path):
                        raise BuildError(f"Built binary does not contain main(): {exe_path}")

                    binary = exe_path
                    if target == TargetType.OPENCL and build_plan.kernels:
                        # Pick a kernel from the *project*, never from backups/artifact dirs.
                        kernel_rel = None
                        for k in build_plan.kernels:
                            if not k or k.startswith("."):
                                continue
                            if "nnport" in k.lower() or "backup" in k.lower():
                                continue
                            kernel_rel = k
                            break
                        if kernel_rel:
                            k_abs = os.path.join(project_root, kernel_rel)
                            if os.path.exists(k_abs):
                                binary = (exe_path, k_abs)
                else:
                    toolchain = config.toolchain if hasattr(config, 'toolchain') else None
                    binary = self.compiler.compile(
                        host_code,
                        target,
                        config.compiler_cmd,
                        toolchain_config=toolchain,
                        kernel_code=current_kernel,
                        extra_include_paths=extra_includes,
                    )
                    if kernel_path_for_run and isinstance(binary, str):
                        binary = (binary, kernel_path_for_run)
                add_log({"stage": f"Iteration {iteration}", "status": "Compilation Success", "details": f"Binary: {binary}"})
            except Exception as e:
                compilation_error = str(e)
                add_log({"stage": f"Iteration {iteration}", "status": "Compilation Failed", "details": compilation_error})
                classified = classify_compilation_error(compilation_error)
                compilation_history.append(classified.normalized_signature)

                # Stop early if we are repeating the same compile error (infinite loop prevention).
                if len(compilation_history) >= 3:
                    last_three = compilation_history[-3:]
                    if all(sig == last_three[0] for sig in last_three):
                        add_log(
                            {
                                "stage": "Result",
                                "status": "Failed",
                                "details": f"⚠️ STUCK IN LOOP: Same compilation error repeated 3 times.\n\nError: {last_three[0]}\n\nThis is likely an environment/dependency issue (headers/toolchain), not a code issue.",
                            }
                        )
                        return logs

                # Stage-gated handling for common non-code failures.
                if classified.failure_class == FailureClass.TOOLCHAIN:
                    add_log({"stage": classified.stage, "status": "Failed", "details": classified.hint or "Toolchain failure"})
                    return logs
                if classified.failure_class == FailureClass.DEPENDENCY:
                    add_log({"stage": classified.stage, "status": "Failed", "details": classified.hint or "Dependency failure"})
                    return logs
                if classified.failure_class == FailureClass.BUILD_PLAN:
                    # Re-analyze and retry next iteration (no LLM).
                    if project_root and project_context:
                        try:
                            project_profile = analyze_project(project_root)
                        except Exception:
                            pass
                    continue

                # Stage-gated stop conditions for non-code failures (toolchain/dependencies).
                # These are not reliably solvable by editing source code.
                if "Android NDK compiler not found" in compilation_error or "NDK compiler not found" in compilation_error:
                    add_log(
                        {
                            "stage": "Toolchain Validation",
                            "status": "Failed",
                            "details": "Android NDK toolchain is missing or misconfigured. Fix NDK installation/path, then retry.",
                        }
                    )
                    return logs

                if "Compiler " in compilation_error and "not found" in compilation_error:
                    add_log(
                        {
                            "stage": "Toolchain Validation",
                            "status": "Failed",
                            "details": f"Compiler is missing on host. Error: {compilation_error[:200]}",
                        }
                    )
                    return logs

                if target == TargetType.OPENCL and ("CL/cl.h" in compilation_error or "opencl.h" in compilation_error) and "file not found" in compilation_error:
                    vendored = os.path.join(os.path.dirname(__file__), "opencl_sdk", "include", "CL", "cl.h")
                    if not os.path.exists(vendored):
                        add_log(
                            {
                                "stage": "Dependency Resolution",
                                "status": "Failed",
                                "details": "OpenCL headers are missing. For Android cross-compiles, the NDK does not ship OpenCL headers; NNPort must provide them (vendored headers + -I include path).",
                            }
                        )
                        return logs
                
                # If reference exists and not last iteration, try AI fix
                if ref_output is not None and iteration < max_iterations - 1:
                    add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": "Using AI to fix compilation errors..."})
                    try:
                        # Ask AI to fix the compilation error
                        with open(reference_model_path, 'r') as f:
                            ref_content = f.read()
                        
                        error_feedback = f"COMPILATION ERROR:\n{compilation_error}\n\nFix the project so it compiles successfully."
                        if project_root and project_context:
                            ops_obj = self.generator.generate(
                                ref_content,
                                target,
                                iteration,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                                output_format="json_ops",
                            )
                            run_id = f"{job_id or 'manual'}_{iteration}_{uuid.uuid4().hex[:8]}"
                            ws = get_nnport_workspace(project_root, create=True)
                            apply_ops_in_place(project_root, ops_obj, run_id=run_id, backups_root=ws.backups_root)
                            # Reload project context after applying edits
                            try:
                                refreshed_files = {}
                                for relp in (project_context.get("files") or {}).keys():
                                    abs_p = os.path.join(project_root, relp)
                                    if os.path.exists(abs_p):
                                        try:
                                            with open(abs_p, "r") as pf:
                                                refreshed_files[relp] = pf.read()
                                        except Exception:
                                            pass
                                project_context["files"] = refreshed_files
                            except Exception:
                                pass
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": f"Applied JSON ops to project (backup: .nnport_backups/{run_id})"})
                        else:
                            current_code = self.generator.generate(
                                ref_content,
                                target,
                                iteration,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                            )
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated new code version"})
                        continue
                    except Exception as fix_error:
                        add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
                        continue
                else:
                    continue
            
            # Run on TARGET - ALWAYS REQUIRED, NO MOCK MODE
            if not (config.connection_type and (config.connection_type == "adb" or config.connection_type == "ssh" or config.connection_type == "local")):
                add_log({"stage": "Result", "status": "Failed", "details": "❌ NO DEPLOYMENT METHOD CONFIGURED!\n\nYou MUST configure a connection method:\n- ADB (Android device)\n- SSH (remote server)\n- Local (compile and run on host)\n\nGo to device settings and select a connection type. NO MOCK EXECUTION ALLOWED!"})
                return logs
            
            add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Deploying", "details": f"Pushing to {config.connection_type} device..."})
            try:
                actual_output = self.runner.deploy_and_run(binary, input_data, config, expected_output=ref_output, logs=logs)
                add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Complete", "details": f"Output shape: {actual_output.shape} | Values: {actual_output.flatten()[:5].tolist()}"})
            except Exception as e:
                execution_error = str(e)
                classified = classify_runtime_error(execution_error)
                # Track error for loop detection (normalized to remove volatile ids/paths)
                error_signature = classified.normalized_signature
                error_history.append(error_signature)
                
                add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Execution Failed", "details": execution_error})
                # Print to console for debugging
                print(f"❌ Execution error on iteration {iteration}: {execution_error}")

                # Stage-gated handling for common non-code failures.
                if classified.failure_class == FailureClass.BUILD_PLAN:
                    add_log(
                        {
                            "stage": classified.stage,
                            "status": "Failed",
                            "details": f"{classified.hint or 'Build plan failure'}\n\n{classified.normalized_signature}",
                        }
                    )
                    # Re-analyze and rebuild plan next iteration (no LLM).
                    if project_root and project_context:
                        try:
                            project_profile = analyze_project(project_root)
                        except Exception:
                            pass
                    continue
                if classified.failure_class == FailureClass.DEPLOY:
                    add_log({"stage": "Result", "status": "Failed", "details": classified.hint or "Deployment error"})
                    return logs
                if classified.failure_class == FailureClass.DEPENDENCY and "OpenCL" in (classified.hint or ""):
                    add_log({"stage": "Result", "status": "Failed", "details": classified.hint or "Dependency/runtime linking error"})
                    return logs
                
                # IMMEDIATE loop detection - check right after adding error
                if len(error_history) >= 3:
                    last_three = error_history[-3:]
                    if all(e == last_three[0] for e in last_three):
                        # Only stop early for non-code failures. For code-level issues (like OpenCL -52),
                        # keep iterating until max_iterations is reached.
                        if classified.failure_class in (FailureClass.DEPLOY, FailureClass.DEPENDENCY, FailureClass.BUILD_PLAN):
                            add_log(
                                {
                                    "stage": "Result",
                                    "status": "Failed",
                                    "details": f"⚠️ STUCK IN LOOP: Same error repeated 3 times.\n\nError: {error_signature}\n\nHint: {classified.hint or 'n/a'}",
                                }
                            )
                            return logs
                
                # Connection/deployment errors are not fixable by code edits.
                if "No Android devices connected" in execution_error or "ADB not found" in execution_error or "SSH connection" in execution_error:
                    add_log({"stage": "Result", "status": "Failed", "details": "Deployment error - cannot reach target device. Fix connection settings."})
                    return logs
                
                # Try AI fix if not last iteration and it's a code/runtime error
                if ref_output is not None and iteration < max_iterations - 1:
                    add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": "Asking AI to fix execution error..."})
                    try:
                        with open(reference_model_path, 'r') as f:
                            ref_content = f.read()
                        
                        # Enhanced error feedback with specific guidance
                        error_guidance = "Fix the code to execute successfully on the target device."
                        
                        # Specific guidance for common OpenCL errors
                        if "cannot locate symbol" in execution_error and "cl" in execution_error.lower():
                            error_guidance += "\n\nOPENCL LINKING ERROR DETECTED:\nThe OpenCL library (libOpenCL.so) is not being found at runtime.\nNote: The library should already be pushed to the device at /data/local/tmp/libOpenCL.so.\n\nPossible fixes:\n1. Check that the wrapper script is using LD_LIBRARY_PATH=.\n2. Verify libOpenCL.so is in the same directory as the binary\n3. Check that all OpenCL API calls use correct function signatures\n4. Ensure you're not using deprecated OpenCL 1.x functions"
                        elif "kernel" in execution_error.lower() and ("not found" in execution_error or "failed" in execution_error):
                            error_guidance += "\n\nKERNEL ERROR DETECTED:\n1. Make sure kernel.cl is being read correctly from the same directory\n2. Check kernel function name matches the clCreateKernel() call\n3. Verify kernel syntax is valid OpenCL C"
                        
                        error_feedback = f"EXECUTION ERROR:\n{execution_error}\n\n{error_guidance}"
                        if project_root and project_context:
                            ops_obj = self.generator.generate(
                                ref_content,
                                target,
                                iteration + 1,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                                output_format="json_ops",
                            )
                            run_id = f"{job_id or 'manual'}_exec_{iteration}_{uuid.uuid4().hex[:8]}"
                            ws = get_nnport_workspace(project_root, create=True)
                            try:
                                res = apply_ops_in_place(project_root, ops_obj, run_id=run_id, backups_root=ws.backups_root)
                            except OpsValidationError as oe:
                                # If SHA precondition failed, retry once by stripping expected_sha256_before fields.
                                if "SHA mismatch" in str(oe):
                                    try:
                                        if isinstance(ops_obj, dict) and isinstance(ops_obj.get("ops"), list):
                                            for op in ops_obj["ops"]:
                                                if isinstance(op, dict):
                                                    op.pop("expected_sha256_before", None)
                                        res = apply_ops_in_place(project_root, ops_obj, run_id=run_id + "_retry", backups_root=ws.backups_root)
                                        run_id = run_id + "_retry"
                                    except Exception:
                                        raise
                                else:
                                    raise
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": f"Applied JSON ops to project (backup: {res.get('backup_root', run_id)})"})
                        else:
                            current_code = self.generator.generate(
                                ref_content,
                                target,
                                iteration + 1,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                            )
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated new code to fix execution error"})
                        continue
                    except Exception as fix_error:
                        add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
                continue
            
            # Compare with reference if available
            if ref_output is not None:
                add_log({"stage": f"Iteration {iteration}", "status": "Comparing", "details": f"HOST shape: {ref_output.shape} | TARGET shape: {actual_output.shape}"})
                
                # Check shape match
                if ref_output.shape != actual_output.shape:
                    shape_error = f"Expected {ref_output.shape}, got {actual_output.shape}"
                    add_log({"stage": f"Iteration {iteration}", "status": "Shape Mismatch", "details": shape_error})
                    
                    # Try AI fix if not last iteration
                    if iteration < max_iterations - 1:
                        add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": "Asking AI to fix shape mismatch..."})
                        try:
                            with open(reference_model_path, 'r') as f:
                                ref_content = f.read()
                            
                            # Generate new code with error feedback
                            error_feedback = f"OUTPUT SHAPE MISMATCH:\nExpected output shape: {ref_output.shape}\nActual output shape: {actual_output.shape}\n\nThe C code is not producing the correct output shape. Make sure:\n1. You're writing ALL output values to output.bin (not just the first element)\n2. The fwrite() call writes the complete array: fwrite(output_array, sizeof(float), TOTAL_OUTPUT_SIZE, fp)\n3. TOTAL_OUTPUT_SIZE should be the product of all output dimensions"
                            if project_root and project_context:
                                ops_obj = self.generator.generate(
                                    ref_content,
                                    target,
                                    iteration + 1,
                                    error_feedback=error_feedback,
                                    debug_instructions=debug_instructions,
                                    device_config=config,
                                    project_context=project_context,
                                    output_format="json_ops",
                                )
                                run_id = f"{job_id or 'manual'}_shape_{iteration}_{uuid.uuid4().hex[:8]}"
                                ws = get_nnport_workspace(project_root, create=True)
                                apply_ops_in_place(project_root, ops_obj, run_id=run_id, backups_root=ws.backups_root)
                                add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": f"Applied JSON ops to project (backup: .nnport_backups/{run_id})"})
                            else:
                                current_code = self.generator.generate(
                                    ref_content,
                                    target,
                                    iteration + 1,
                                    error_feedback=error_feedback,
                                    debug_instructions=debug_instructions,
                                    device_config=config,
                                    project_context=project_context,
                                )
                                # Extract preview for logging
                                code_preview = current_code.get("host_code", "")[:500] if isinstance(current_code, dict) else current_code[:500]
                                add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated new code to fix shape", "source_preview": code_preview + "..." if len(code_preview) >= 500 else code_preview})
                            continue
                        except Exception as fix_error:
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
                            continue
                    else:
                        add_log({"stage": "Result", "status": "Failed", "details": f"Shape mismatch after {max_iterations} iterations"})
                        break
                
                # Compute difference
                diff = np.linalg.norm(ref_output - actual_output)
                add_log({"stage": f"Iteration {iteration}", "status": "Verified", "details": f"L2 Diff: {diff:.4f} | Threshold: 1e-4"})
                
                best_diff = min(best_diff, diff)
                
                if diff < 1e-4:
                    # Promote a commit-friendly build file into the project root (B-only policy).
                    if project_root and project_profile:
                        try:
                            add_log({"stage": "Project Promotion", "status": "Running", "details": "Writing CMakeLists.txt into project root..."})
                            ws = get_nnport_workspace(project_root, create=True)
                            ops_obj = promotion_ops(project_profile, target_exe="nnport_app")
                            apply_ops_in_place(project_root, ops_obj, run_id=f"{job_id or 'manual'}_promote_{iteration}", backups_root=ws.backups_root)
                            add_log({"stage": "Project Promotion", "status": "Success", "details": "Updated/created CMakeLists.txt in project root"})
                        except Exception as e:
                            add_log({"stage": "Project Promotion", "status": "Failed", "details": str(e)})
                    add_log({"stage": "Result", "status": "Success", "details": f"Code matches reference! Converged in iteration {iteration}"})
                    return logs
                elif iteration < max_iterations - 1:
                    # Try AI fix for value mismatch
                    add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": f"L2 diff too high ({diff:.4f}), asking AI to improve..."})
                    try:
                        with open(reference_model_path, 'r') as f:
                            ref_content = f.read()
                        
                        error_feedback = f"OUTPUT VALUE MISMATCH:\nL2 norm difference: {diff:.4f} (threshold: 1e-4)\nExpected output: {ref_output.flatten()[:10]}\nActual output: {actual_output.flatten()[:10]}\n\nThe computation is producing incorrect values. Review the model logic and ensure all operations match the PyTorch reference exactly."
                        if project_root and project_context:
                            ops_obj = self.generator.generate(
                                ref_content,
                                target,
                                iteration + 1,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                                output_format="json_ops",
                            )
                            run_id = f"{job_id or 'manual'}_value_{iteration}_{uuid.uuid4().hex[:8]}"
                            ws = get_nnport_workspace(project_root, create=True)
                            apply_ops_in_place(project_root, ops_obj, run_id=run_id, backups_root=ws.backups_root)
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": f"Applied JSON ops to project (backup: .nnport_backups/{run_id})"})
                        else:
                            current_code = self.generator.generate(
                                ref_content,
                                target,
                                iteration + 1,
                                error_feedback=error_feedback,
                                debug_instructions=debug_instructions,
                                device_config=config,
                                project_context=project_context,
                            )
                            # Extract preview for logging
                            code_preview = current_code.get("host_code", "")[:500] if isinstance(current_code, dict) else current_code[:500]
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated improved code version", "source_preview": code_preview + "..." if len(code_preview) >= 500 else code_preview})
                    except Exception as fix_error:
                        add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
            else:
                # No reference, just report success
                if project_root and project_profile:
                    try:
                        add_log({"stage": "Project Promotion", "status": "Running", "details": "Writing CMakeLists.txt into project root..."})
                        ws = get_nnport_workspace(project_root, create=True)
                        ops_obj = promotion_ops(project_profile, target_exe="nnport_app")
                        apply_ops_in_place(project_root, ops_obj, run_id=f"{job_id or 'manual'}_promote_noref_{iteration}", backups_root=ws.backups_root)
                        add_log({"stage": "Project Promotion", "status": "Success", "details": "Updated/created CMakeLists.txt in project root"})
                    except Exception as e:
                        add_log({"stage": "Project Promotion", "status": "Failed", "details": str(e)})
                add_log({"stage": "Result", "status": "Success", "details": "Code executed successfully (no reference comparison)"})
                return logs
        
        # After all iterations
        if ref_output is not None:
            add_log({"stage": "Result", "status": "Failed", "details": f"Could not match reference after {max_iterations} iterations. Best L2 diff: {best_diff:.4f}"})
        
        return logs
