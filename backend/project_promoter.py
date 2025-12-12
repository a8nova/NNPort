"""Project promotion (B-only): write build files into project root.

This makes the project commit-friendly by materializing a real project-level
`CMakeLists.txt` that builds the currently-selected sources using relative paths.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from backend.project_analyzer import ProjectProfile


def _rel_sources(profile: ProjectProfile) -> List[str]:
    # Only include real project sources (profile already prunes nnport artifacts).
    sources = sorted(set(profile.sources_cpp + profile.sources_c))
    # Prefer entrypoint first.
    if profile.selected_entrypoint and profile.selected_entrypoint in sources:
        sources.remove(profile.selected_entrypoint)
        sources = [profile.selected_entrypoint] + sources
    return sources


def _rel_include_dirs(profile: ProjectProfile) -> List[str]:
    dirs = set(["."])
    for h in profile.headers:
        d = os.path.dirname(h)
        dirs.add(d if d else ".")
    return sorted(dirs)


def generate_project_cmakelists(profile: ProjectProfile, *, target_exe: str = "nnport_app") -> str:
    sources = _rel_sources(profile)
    include_dirs = _rel_include_dirs(profile)

    src_lines = "\n".join([f"  {s}" for s in sources])
    inc_lines = "\n".join([f"target_include_directories({target_exe} PRIVATE {d})" for d in include_dirs])

    return f"""cmake_minimum_required(VERSION 3.16)
project(nnport_project LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable({target_exe}
{src_lines}
)

{inc_lines}

# NOTE:
# - For Android builds, configure with the NDK toolchain file:
#   -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
#   -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21
#
# - OpenCL symbols are typically resolved at runtime on Android devices.
if(ANDROID)
  target_link_options({target_exe} PRIVATE
    -Wl,--allow-shlib-undefined
    -Wl,--unresolved-symbols=ignore-all
  )
endif()
"""


def promotion_ops(profile: ProjectProfile, *, target_exe: str = "nnport_app") -> Dict[str, Any]:
    """Return JSON ops to promote build files into project root."""
    cmake = generate_project_cmakelists(profile, target_exe=target_exe)
    return {
        "ops": [
            {
                "type": "update" if os.path.exists(os.path.join(profile.project_root, "CMakeLists.txt")) else "create",
                "path": "CMakeLists.txt",
                "content": cmake,
            }
        ]
    }

