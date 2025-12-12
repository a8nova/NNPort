"""Failure classification and signature normalization.

Purpose: stop wasting iterations by recognizing repeated failures and routing to the
right kind of fix (toolchain/deps/build plan/code/deploy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureClass(str, Enum):
    TOOLCHAIN = "toolchain"
    DEPENDENCY = "dependency"
    BUILD_PLAN = "build_plan"
    CODE = "code"
    DEPLOY = "deploy"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedFailure:
    failure_class: FailureClass
    stage: str
    normalized_signature: str
    raw: str
    hint: Optional[str] = None


_BINARY_RE = re.compile(r"(?:\./)?binary_[0-9a-f]{8,}\.bin")
_PATH_RE = re.compile(r"/[^\s]+")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def normalize_signature(msg: str, max_len: int = 160) -> str:
    s = msg or ""
    # Remove volatile binary names
    s = _BINARY_RE.sub("binary_<id>.bin", s)
    s = s.replace("./binary_<id>.bin", "binary_<id>.bin")
    # Strip run UUIDs (e.g., backup folder ids)
    s = _UUID_RE.sub("<uuid>", s)
    # Collapse absolute paths (keep last path segment)
    def _path_repl(m):
        p = m.group(0)
        base = p.rstrip("/").split("/")[-1]
        return f"/<path>/{base}" if base else "/<path>"

    s = _PATH_RE.sub(_path_repl, s)
    s = " ".join(s.split())
    return s[:max_len]


def classify_compilation_error(err: str) -> ClassifiedFailure:
    sig = normalize_signature(err)

    if "NDK" in err and "not found" in err:
        return ClassifiedFailure(FailureClass.TOOLCHAIN, "Toolchain Validation", sig, err, "Android NDK missing/misconfigured")

    if "file not found" in err and ("CL/cl.h" in err or "opencl.h" in err):
        return ClassifiedFailure(FailureClass.DEPENDENCY, "Dependency Resolution", sig, err, "OpenCL headers missing")

    if "cannot locate symbol" in err and "main" in err:
        return ClassifiedFailure(FailureClass.BUILD_PLAN, "Build/Entrypoint", sig, err, "Entrypoint not linked (main missing)")

    return ClassifiedFailure(FailureClass.UNKNOWN, "Compile", sig, err)


def classify_runtime_error(err: str) -> ClassifiedFailure:
    sig = normalize_signature(err)

    if "No Android devices connected" in err or "ADB not found" in err:
        return ClassifiedFailure(FailureClass.DEPLOY, "Deploy", sig, err, "ADB connectivity")

    if "cannot locate symbol" in err and "main" in err:
        return ClassifiedFailure(FailureClass.BUILD_PLAN, "Build/Entrypoint", sig, err, "Entrypoint not linked (main missing)")

    if "cannot locate symbol" in err and "cl" in err.lower():
        return ClassifiedFailure(FailureClass.DEPENDENCY, "Runtime Linking", sig, err, "OpenCL runtime symbol resolution")

    # OpenCL runtime error codes that are usually fixable in generated code.
    # Example seen in logs: clEnqueueNDRangeKernel -> CL_INVALID_KERNEL_ARGS (-52)
    if "Failed to execute kernel" in err and "code -52" in err:
        return ClassifiedFailure(
            FailureClass.CODE,
            "OpenCL Kernel Args",
            sig,
            err,
            "OpenCL error -52 (CL_INVALID_KERNEL_ARGS): kernel arg mismatch (types/sizes/order). Fix host clSetKernelArg calls and kernel signature.",
        )
    if "Failed to execute kernel" in err and "code -" in err:
        return ClassifiedFailure(
            FailureClass.CODE,
            "OpenCL Kernel Launch",
            sig,
            err,
            "OpenCL kernel launch failed: likely argument mismatch, invalid global size, or kernel signature mismatch.",
        )

    return ClassifiedFailure(FailureClass.RUNTIME, "Run", sig, err)

