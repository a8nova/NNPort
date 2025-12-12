"""Build planning utilities.

You selected the policy: generate a fresh build system under `.nnport/`.

This module creates an NNPort-owned CMake harness that builds a single executable
from discovered sources. (CMake detection is supported first.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from backend.project_analyzer import ProjectProfile
from backend.nnport_workspace import get_nnport_workspace


@dataclass
class BuildPlan:
    project_root: str
    mode: str  # "cmake_harness"
    entrypoint: str
    sources: List[str]
    include_dirs: List[str]
    kernels: List[str]
    run_dir: str
    harness_dir: str
    build_dir: str
    toolchain_file: Optional[str]
    cmake_defines: Dict[str, str]
    target_exe: str

    def to_dict(self) -> Dict:
        return asdict(self)

CMAKELISTS_TEMPLATE = r"""cmake_minimum_required(VERSION 3.16)
project(nnport_harness LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable({target_exe}
{sources}
)

# Include dirs
{include_dirs}

# Linker flags: allow unresolved OpenCL symbols for Android runtime resolution
target_link_options({target_exe} PRIVATE
  -Wl,--allow-shlib-undefined
  -Wl,--unresolved-symbols=ignore-all
)

# Use OpenCL headers include root if present
"""


def generate_cmake_harness(
    profile: ProjectProfile,
    *,
    target_exe: str = "nnport_app",
    ndk_path: Optional[str] = None,
    android_abi: str = "arm64-v8a",
    android_api: str = "21",
    opencl_include_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> BuildPlan:
    project_root = profile.project_root

    if not profile.selected_entrypoint:
        raise RuntimeError("No main() entrypoint detected; cannot generate build harness")

    # Build source list: all C/C++ sources. (Heuristic; refined later.)
    sources = sorted(set(profile.sources_cpp + profile.sources_c))

    # Avoid multiple-main link errors by excluding other main() candidates.
    other_mains = {c.relpath for c in (profile.main_candidates or []) if c.relpath != profile.selected_entrypoint}
    if other_mains:
        sources = [s for s in sources if s not in other_mains]

    # Ensure entrypoint is included and first.
    if profile.selected_entrypoint in sources:
        sources.remove(profile.selected_entrypoint)
    sources = [profile.selected_entrypoint] + sources

    # Include the project root plus header directories inferred from files.
    include_dirs_set = {"."}
    for h in profile.headers:
        d = os.path.dirname(h)
        include_dirs_set.add(d if d else ".")
    if opencl_include_root:
        # This is an absolute path; it will be written into the harness as-is.
        include_dirs_set.add(opencl_include_root)
    include_dirs = sorted(include_dirs_set)

    run_id = run_id or "latest"
    ws = get_nnport_workspace(project_root, create=True)
    run_dir = os.path.join(ws.runs_root, run_id)
    harness_dir = os.path.join(run_dir, "harness")
    build_dir = os.path.join(run_dir, "build")
    os.makedirs(harness_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)

    toolchain_file = None
    cmake_defines: Dict[str, str] = {}
    if ndk_path:
        # Use the official NDK CMake toolchain file (most compatible).
        toolchain_file = os.path.join(ndk_path, "build", "cmake", "android.toolchain.cmake")
        cmake_defines.update(
            {
                "ANDROID_ABI": android_abi,
                "ANDROID_PLATFORM": f"android-{android_api}",
                "ANDROID_STL": "c++_static",
            }
        )

    cmakelists_path = os.path.join(harness_dir, "CMakeLists.txt")

    # Harness layout: harness_dir contains CMakeLists.txt; sources are referenced relative to project_root.
    sources_lines = "".join([f"  \"{os.path.join(project_root, s)}\"\n" for s in sources])
    include_lines = ""
    for d in include_dirs:
        if os.path.isabs(d):
            include_lines += f"target_include_directories({target_exe} PRIVATE \"{d}\")\n"
        else:
            include_lines += f"target_include_directories({target_exe} PRIVATE \"{os.path.join(project_root, d)}\")\n"

    with open(cmakelists_path, "w", encoding="utf-8") as f:
        f.write(
            CMAKELISTS_TEMPLATE.format(
                target_exe=target_exe,
                sources=sources_lines,
                include_dirs=include_lines,
            )
        )

    return BuildPlan(
        project_root=project_root,
        mode="cmake_harness",
        entrypoint=profile.selected_entrypoint,
        sources=sources,
        include_dirs=include_dirs,
        kernels=profile.kernels_cl,
        run_dir=run_dir,
        harness_dir=harness_dir,
        build_dir=build_dir,
        toolchain_file=toolchain_file,
        cmake_defines=cmake_defines,
        target_exe=target_exe,
    )

