"""NNPort workspace location resolver.

Requirement: store generated build artifacts *inside the project root* in a visible,
conventional place, without forcing a new top-level folder name.

Policy:
- Prefer using an existing build directory when present.
- Otherwise create a standard `build/` and place NNPort artifacts under `build/nnport/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class NNPortWorkspace:
    project_root: str
    workspace_root: str  # e.g. <project_root>/build/nnport
    runs_root: str       # <workspace_root>/runs
    backups_root: str    # <workspace_root>/backups


def _first_existing_build_dir(project_root: str) -> str | None:
    # Common conventions
    for name in ("build", "Build", "out", "Out"):
        cand = os.path.join(project_root, name)
        if os.path.isdir(cand):
            return cand
    # CMake default-ish patterns users often have
    try:
        for name in os.listdir(project_root):
            if name.startswith("cmake-build-"):
                cand = os.path.join(project_root, name)
                if os.path.isdir(cand):
                    return cand
    except OSError:
        pass
    return None


def get_nnport_workspace(project_root: str, *, create: bool = True) -> NNPortWorkspace:
    project_root = os.path.realpath(project_root)

    build_root = _first_existing_build_dir(project_root)
    if build_root is None:
        build_root = os.path.join(project_root, "build")
        if create:
            os.makedirs(build_root, exist_ok=True)

    workspace_root = os.path.join(build_root, "nnport")
    runs_root = os.path.join(workspace_root, "runs")
    backups_root = os.path.join(workspace_root, "backups")

    if create:
        os.makedirs(runs_root, exist_ok=True)
        os.makedirs(backups_root, exist_ok=True)

    return NNPortWorkspace(
        project_root=project_root,
        workspace_root=workspace_root,
        runs_root=runs_root,
        backups_root=backups_root,
    )

