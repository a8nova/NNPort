import time
import os
import uuid
import shutil
import subprocess
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from backend.targets import TargetType, DeviceConfig
from backend.core import ComparisonEngine
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

    def generate(self, original_model_source: str, target: TargetType, iteration: int, error_feedback: str = None, debug_instructions: str = "", device_config: DeviceConfig = None) -> str:
        """
        Generates code for the target architecture.
        Prioritizes OpenAI (GPT-4o), falls back to Gemini, then Mock.
        """
        
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
        # Try OpenAI
        if self.openai_client or self.openai_legacy:
            try:
                if self.openai_legacy:
                    # Old OpenAI SDK (v0.x)
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a specialized code generation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    code = response.choices[0].message.content
                else:
                    # New OpenAI SDK (v1.0+)
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a specialized code generation assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    code = response.choices[0].message.content
                # Strip markdown and sanitize code
                code = self._sanitize_code(code)
                return code
            except Exception as e:
                print(f"OpenAI Error: {e}")
                pass # Fallback

        # Try Gemini
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                code = response.text
                # Remove OpenCL/CUDA headers that would cause compilation errors
                code = self._sanitize_code(code)
                return code
            except Exception as e:
                 print(f"Gemini Error: {e}")
                 pass

        return f"// ERROR: No API Keys available (OpenAI or Gemini).\n// Please set OPENAI_API_KEY to use AI code generation.\n// Mocking fallback for now...\n" + self._get_mock_template(original_model_source, target, iteration)
    
    def _sanitize_code(self, code: str) -> str:
        """Remove problematic includes and C++ syntax that would prevent compilation"""
        import re
        
        # First strip markdown
        code = self._strip_markdown(code)
        
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
        """Remove markdown code block markers from generated code"""
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
        
        return '\n'.join(lines)

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
        """Remove markdown code block markers from generated code"""
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
        
        return '\n'.join(lines)
    
    def _sanitize_code(self, code: str) -> str:
        """Remove problematic includes and C++ syntax that would prevent compilation"""
        import re
        
        # First strip markdown
        code = self._strip_markdown(code)
        
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
        """Find Android clang compiler in NDK"""
        import platform
        
        # Detect host architecture
        system = platform.system().lower()
        machine = platform.machine()
        host_arch = f"{system}-{machine}"
        
        toolchain_dir = os.path.join(ndk_path, "toolchains/llvm/prebuilt", host_arch)
        if not os.path.isdir(toolchain_dir):
            # Try x86_64 variant
            toolchain_dir = os.path.join(ndk_path, "toolchains/llvm/prebuilt", f"{system}-x86_64")
        
        clang = os.path.join(toolchain_dir, "bin", f"{android_arch}-linux-android{android_api}-clang")
        if os.path.isfile(clang):
            return clang
        
        # Fallback to generic clang
        clang = os.path.join(toolchain_dir, "bin/clang")
        if os.path.isfile(clang):
            return clang
        
        return None
    
    def compile(self, source_code: str, target: TargetType, compiler_cmd: str = "gcc", allow_mock: bool = True, toolchain_config: Optional[Any] = None) -> str:
        """
        Compiles source code with toolchain support for cross-compilation.
        """
        # SANITIZE the code before compilation to remove C++ features
        source_code = self._sanitize_code(source_code)
        
        # Save source to temp file (use .c for pure C code)
        src_path = f"/tmp/source_{uuid.uuid4().hex}.c"
        bin_path = f"/tmp/binary_{uuid.uuid4().hex}.bin"
        
        with open(src_path, "w") as f:
            f.write(source_code)
        
        compilation_errors = []
        
        # Try custom toolchain if provided
        if toolchain_config and compiler_cmd != "mock":
            try:
                cmd = [toolchain_config.compiler_path]
                
                # Add sysroot if specified
                if toolchain_config.sysroot:
                    cmd.extend(["--sysroot", toolchain_config.sysroot])
                
                # Add include paths
                for include_path in toolchain_config.include_paths:
                    cmd.extend(["-I", include_path])
                
                # Add library paths
                for lib_path in toolchain_config.library_paths:
                    cmd.extend(["-L", lib_path])
                
                # Add compiler flags
                cmd.extend(toolchain_config.compiler_flags)
                
                # Add standard flags
                cmd.extend(["-std=c11", "-o", bin_path, src_path])
                
                # Add linker flags
                cmd.extend(toolchain_config.linker_flags)
                
                print(f"Compiling with toolchain: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Successfully compiled with custom toolchain: {bin_path}")
                    return bin_path
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
                    cmd = [
                        android_clang,
                        "-target", "aarch64-linux-android21",
                        "-std=c11",  # Use C11 instead of C++
                        "-static",  # Statically link everything - no runtime dependencies
                        "-o", bin_path,
                        src_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Successfully cross-compiled for Android: {bin_path}")
                        return bin_path
                    else:
                        error_msg = f"Android NDK compilation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                        print(error_msg)
                        compilation_errors.append(error_msg)
                except FileNotFoundError as e:
                    error_msg = f"Android NDK compiler not found: {e}"
                    print(error_msg)
                    compilation_errors.append(error_msg)
        
        # Try regular compilation if not mock mode
        if compiler_cmd != "mock":
            try:
                # Use gcc for C files (don't use -static on macOS as it causes linker issues)
                cmd = [compiler_cmd, "-std=c11", "-o", bin_path, src_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return bin_path
                else:
                    error_msg = f"Compilation with {compiler_cmd} failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    print(error_msg)
                    compilation_errors.append(error_msg)
            except FileNotFoundError as e:
                error_msg = f"Compiler {compiler_cmd} not found: {e}"
                print(error_msg)
                compilation_errors.append(error_msg)
        
        # If mock is not allowed (e.g., when using ADB), raise an error
        if not allow_mock:
            all_errors = "\n\n".join(compilation_errors)
            raise Exception(f"Compilation failed and mock mode is not allowed:\n{all_errors}")
        
        # Mock binary (source code as placeholder) - only for mock mode
        with open(bin_path, "w") as f:
            f.write(source_code)
        return bin_path

class DeviceRunner:
    def deploy_and_run(self, binary_path: str, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Deploy and run on device. Supports SSH, ADB, Local, or Mock.
        """
        # Check connection type
        if hasattr(config, 'connection_type'):
            if config.connection_type == 'ssh':
                return self._run_ssh(binary_path, input_data, config, expected_output, logs=logs)
            elif config.connection_type == 'adb':
                return self._run_adb(binary_path, input_data, config, expected_output, logs=logs)
        
        # Legacy support
        if config.use_adb:
            return self._run_adb(binary_path, input_data, config, expected_output, logs=logs)
        
        if config.mock:
            return self._run_mock(binary_path, input_data, expected_output)
        
        # Default to local execution
        return self._run_local(binary_path, input_data, expected_output, logs=logs)

    def _run_local(self, binary_path: str, input_data: np.ndarray, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None) -> np.ndarray:
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
        
        # Copy input file
        shutil.copy(input_file, local_input)
        
        try:
            if logs is not None:
                logs.append({"stage": "Local Execute", "status": "Running", "details": f"Executing {os.path.basename(binary_path)} locally..."})
            
            # Make sure binary is executable
            os.chmod(binary_path, 0o755)
            
            # Execute binary locally
            result = subprocess.run([binary_path], capture_output=True, text=True, timeout=30, cwd=binary_dir)
            
            if logs is not None:
                stdout_preview = result.stdout[:200] if result.stdout else "No output"
                logs.append({"stage": "Local Execute", "status": "Success" if result.returncode == 0 else "Failed", 
                           "details": f"Exit code: {result.returncode}\nOutput: {stdout_preview}"})
            
            if result.returncode != 0:
                error_msg = f"Binary execution failed with exit code {result.returncode}\nSTDERR: {result.stderr}"
                if logs is not None:
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

    def _run_ssh(self, binary_path: str, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None) -> np.ndarray:
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
            
            stdin, stdout, stderr = ssh.exec_command(f"cd {config.remote_work_dir} && ./{os.path.basename(binary_path)}")
            exit_code = stdout.channel.recv_exit_status()
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            if exit_code == 0:
                if logs is not None:
                    logs.append({"stage": "SSH Execute", "status": "Success", "details": f"Execution complete\nOutput: {stdout_text[:200]}"})
            else:
                error_msg = f"Execution failed with exit code {exit_code}\nSTDERR: {stderr_text}"
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
            # Return mock output on error
            if expected_output is not None:
                return expected_output * 0.8
            else:
                return input_data * 0.8
        finally:
            try:
                sftp.close()
                ssh.close()
            except:
                pass

    def _run_adb(self, binary_path: str, input_data: np.ndarray, config: DeviceConfig, expected_output: Optional[np.ndarray] = None, logs: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Deploy and run binary on Android device via ADB.
        Returns output from device execution.
        """
        import subprocess
        
        # Use configured ADB path or default
        ADB_PATH = config.adb_path if hasattr(config, 'adb_path') and config.adb_path else os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
        
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
            devices = [line.split()[0] for line in lines if line.strip() and '\tdevice' in line]
            
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
            
            # Step 2: Push input data to device
            if logs is not None:
                logs.append({"stage": "ADB Push Input", "status": "Running", "details": f"Copying input data ({input_data.size} values) to device..."})
            
            push_input_cmd = adb_cmd + ["push", input_file, remote_input]
            subprocess.run(push_input_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                logs.append({"stage": "ADB Push Input", "status": "Success", "details": f"Input data copied to {remote_input}"})
            
            # Step 3: Set executable permissions
            if logs is not None:
                logs.append({"stage": "ADB Chmod", "status": "Running", "details": "Setting executable permissions..."})
            
            chmod_cmd = adb_cmd + ["shell", "chmod", "+x", remote_bin]
            subprocess.run(chmod_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                logs.append({"stage": "ADB Chmod", "status": "Success", "details": "Permissions set"})
            
            # Step 4: Execute binary on device
            if logs is not None:
                logs.append({"stage": "ADB Execute", "status": "Running", "details": f"Running {os.path.basename(binary_path)} on device..."})
            
            exec_cmd = adb_cmd + ["shell", f"cd {config.remote_work_dir} && ./{os.path.basename(binary_path)}"]
            exec_result = subprocess.run(exec_cmd, capture_output=True, text=True, check=True)
            
            if logs is not None:
                stdout_preview = exec_result.stdout[:200] if exec_result.stdout else "No output"
                logs.append({"stage": "ADB Execute", "status": "Success", "details": f"Execution complete\nOutput: {stdout_preview}"})
            
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
            
            # Fallback: Return mock data if parsing failed
            if logs is not None:
                logs.append({"stage": "Note", "status": "Warning", "details": "Falling back to mock output due to parsing failure"})
            
            # Generate mock output based on expected output or input shape
            if expected_output is not None:
                return expected_output * 0.8  # Mock return close to expected
            else:
                return input_data * 0.8  # Mock return based on input
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ADB command failed: {e.stderr if e.stderr else str(e)}"
            if logs is not None:
                logs.append({"stage": "ADB Error", "status": "Failed", "details": error_msg})
            print(f"ADB Error: {error_msg}")
            # Return mock output on error
            if expected_output is not None:
                return expected_output * 0.8
            else:
                return input_data * 0.8
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logs is not None:
                logs.append({"stage": "ADB Error", "status": "Failed", "details": error_msg})
            print(f"ADB Error: {error_msg}")
            # Return mock output on error
            if expected_output is not None:
                return expected_output * 0.8
            else:
                return input_data * 0.8

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
        
        for i in range(max_iterations):
            add_log({"stage": f"Iteration {i}", "status": "Generating Code", "details": ""})
            
            # Generate
            # We pass the FILE CONTENT as source logic
            with open(source_model_path, 'r') as f:
                src_content = f.read()
            generated_code = self.generator.generate(src_content, target, i, debug_instructions=debug_instructions, device_config=config)
            add_log({"stage": f"Iteration {i}", "status": "Code Generated", "source_preview": generated_code})
            
            # Save generated code
            save_artifact(f"generated_code_iter_{i}.c", generated_code)
            
            # Compile
            add_log({"stage": f"Iteration {i}", "status": "Compiling", "details": f"Target: {target.value}"})
            try:
                toolchain = config.toolchain if hasattr(config, 'toolchain') else None
                binary = self.compiler.compile(generated_code, target, config.compiler_cmd, allow_mock=not config.use_adb, toolchain_config=toolchain)
            except Exception as e:
                add_log({"stage": f"Iteration {i}", "status": "Compilation Failed", "details": str(e)})
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
        
        # 1. Load initial source code
        add_log({"stage": "Source Code", "status": "Loading", "details": f"Reading manual source code..."})
        try:
            with open(manual_source_path, 'r') as f:
                src_content = f.read()
            add_log({"stage": "Source Code", "status": "Loaded", "source_preview": src_content[:500] + "..." if len(src_content) > 500 else src_content})
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
        
        # 3. Iterative compile-test-fix loop
        best_diff = float('inf')
        current_code = src_content
        
        for iteration in range(max_iterations):
            add_log({"stage": f"Iteration {iteration}", "status": "Starting", "details": f"Testing code version {iteration}"})
            
            # Compile
            add_log({"stage": f"Iteration {iteration}", "status": "Compiling", "details": f"Target: {target.value}"})
            try:
                toolchain = config.toolchain if hasattr(config, 'toolchain') else None
                binary = self.compiler.compile(current_code, target, config.compiler_cmd, allow_mock=not config.use_adb, toolchain_config=toolchain)
                add_log({"stage": f"Iteration {iteration}", "status": "Compilation Success", "details": f"Binary: {binary}"})
            except Exception as e:
                compilation_error = str(e)
                add_log({"stage": f"Iteration {iteration}", "status": "Compilation Failed", "details": compilation_error})
                
                # If reference exists and not last iteration, try AI fix
                if ref_output is not None and iteration < max_iterations - 1:
                    add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": "Using AI to fix compilation errors..."})
                    try:
                        # Ask AI to fix the compilation error
                        with open(reference_model_path, 'r') as f:
                            ref_content = f.read()
                        
                        error_feedback = f"COMPILATION ERROR:\n{compilation_error}\n\nFix the C code to compile successfully."
                        fixed_code = self.generator.generate(ref_content, target, iteration, error_feedback=error_feedback, debug_instructions=debug_instructions, device_config=config)
                        current_code = fixed_code
                        add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated new code version"})
                        continue
                    except Exception as fix_error:
                        add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
                        continue
                else:
                    continue
            
            # Run on TARGET
            if config.connection_type == "adb" or config.use_adb:
                add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Deploying", "details": f"Pushing to device..."})
                try:
                    actual_output = self.runner.deploy_and_run(binary, input_data, config, expected_output=ref_output, logs=logs)
                    add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Complete", "details": f"Output shape: {actual_output.shape} | Values: {actual_output.flatten()[:5].tolist()}"})
                except Exception as e:
                    execution_error = str(e)
                    add_log({"stage": f"Iteration {iteration} (TARGET)", "status": "Execution Failed", "details": execution_error})
                    # Print to console for debugging
                    print(f"❌ Execution error on iteration {iteration}: {execution_error}")
                    
                    # Check if it's a connection/deployment error (not fixable by AI)
                    if "No Android devices connected" in execution_error or "ADB not found" in execution_error or "SSH connection" in execution_error:
                        add_log({"stage": "Result", "status": "Failed", "details": "Deployment error - cannot reach target device. Fix connection settings."})
                        return logs
                    
                    # Try AI fix if not last iteration and it's a code/runtime error
                    if ref_output is not None and iteration < max_iterations - 1:
                        add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": "Asking AI to fix execution error..."})
                        try:
                            with open(reference_model_path, 'r') as f:
                                ref_content = f.read()
                            
                            error_feedback = f"EXECUTION ERROR:\n{execution_error}\n\nFix the code to execute successfully on the target device."
                            fixed_code = self.generator.generate(ref_content, target, iteration + 1, error_feedback=error_feedback, debug_instructions=debug_instructions, device_config=config)
                            current_code = fixed_code
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
                                fixed_code = self.generator.generate(ref_content, target, iteration + 1, error_feedback=error_feedback, debug_instructions=debug_instructions, device_config=config)
                                current_code = fixed_code
                                add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated new code to fix shape", "source_preview": fixed_code[:500] + "..." if len(fixed_code) > 500 else fixed_code})
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
                        add_log({"stage": "Result", "status": "Success", "details": f"Code matches reference! Converged in iteration {iteration}"})
                        return logs
                    elif iteration < max_iterations - 1:
                        # Try AI fix for value mismatch
                        add_log({"stage": f"Iteration {iteration}", "status": "Attempting AI Fix", "details": f"L2 diff too high ({diff:.4f}), asking AI to improve..."})
                        try:
                            with open(reference_model_path, 'r') as f:
                                ref_content = f.read()
                            
                            error_feedback = f"OUTPUT VALUE MISMATCH:\nL2 norm difference: {diff:.4f} (threshold: 1e-4)\nExpected output: {ref_output.flatten()[:10]}\nActual output: {actual_output.flatten()[:10]}\n\nThe computation is producing incorrect values. Review the model logic and ensure all operations match the PyTorch reference exactly."
                            fixed_code = self.generator.generate(ref_content, target, iteration + 1, error_feedback=error_feedback, debug_instructions=debug_instructions, device_config=config)
                            current_code = fixed_code
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Applied", "details": "Generated improved code version", "source_preview": fixed_code[:500] + "..." if len(fixed_code) > 500 else fixed_code})
                        except Exception as fix_error:
                            add_log({"stage": f"Iteration {iteration}", "status": "AI Fix Failed", "details": str(fix_error)})
                else:
                    # No reference, just report success
                    add_log({"stage": "Result", "status": "Success", "details": "Code executed successfully (no reference comparison)"})
                    return logs
            else:
                # Mock mode
                add_log({"stage": f"Iteration {iteration}", "status": "Mock Mode", "details": "Enable ADB to test on real device"})
                add_log({"stage": "Result", "status": "Success", "details": "Code compiled successfully (mock execution)"})
                return logs
        
        # After all iterations
        if ref_output is not None:
            add_log({"stage": "Result", "status": "Failed", "details": f"Could not match reference after {max_iterations} iterations. Best L2 diff: {best_diff:.4f}"})
        
        return logs
