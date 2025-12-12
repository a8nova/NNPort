"""Build execution for NNPort build plans."""

from __future__ import annotations

import os
import subprocess
import re
from typing import Tuple

from backend.build_planner import BuildPlan


class BuildError(Exception):
    pass


def _run(cmd, cwd: str, env=None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def build_cmake_harness(plan: BuildPlan) -> str:
    """Configure + build the generated CMake harness.

    Returns: path to produced executable.
    """
    build_dir = plan.build_dir
    harness_dir = plan.harness_dir

    cmake_cmd = ["cmake", "-S", harness_dir, "-B", build_dir]
    if plan.toolchain_file:
        cmake_cmd.extend([f"-DCMAKE_TOOLCHAIN_FILE={plan.toolchain_file}"])

    for k, v in (plan.cmake_defines or {}).items():
        cmake_cmd.append(f"-D{k}={v}")

    rc, out, err = _run(cmake_cmd, cwd=build_dir)
    if rc != 0:
        raise BuildError(f"CMake configure failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")

    rc, out, err = _run(["cmake", "--build", build_dir, "--config", "Release"], cwd=build_dir)
    if rc != 0:
        raise BuildError(f"CMake build failed\nSTDOUT:\n{out}\nSTDERR:\n{err}")

    exe = os.path.join(build_dir, plan.target_exe)
    if os.name == "nt":
        exe += ".exe"
    if not os.path.exists(exe):
        # Some generators place the output under subdirs; do a small check.
        cand = os.path.join(build_dir, "Release", plan.target_exe)
        if os.path.exists(cand):
            exe = cand
        else:
            raise BuildError(f"Build succeeded but executable not found at {exe}")

    return exe


def elf_has_main(binary_path: str) -> bool:
    """Check if the ELF exports/defines a main symbol (best-effort)."""
    try:
        p = subprocess.run(["nm", "-a", binary_path], capture_output=True, text=True)
        if p.returncode == 0 and " main" in p.stdout:
            return True
    except Exception:
        pass

    try:
        p = subprocess.run(["readelf", "-s", binary_path], capture_output=True, text=True)
        if p.returncode == 0 and re.search(r"\bmain\b", p.stdout):
            return True
    except Exception:
        pass

    return False

