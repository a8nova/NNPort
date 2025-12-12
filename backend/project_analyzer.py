"""Project analysis utilities.

Goal: make NNPort behave like an embedded engineer by inspecting an uploaded folder
and producing a build-oriented profile (sources, entrypoints, includes, build files).

This is intentionally lightweight and heuristic-driven; the build planner can refine it.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


_MAIN_RE = re.compile(r"\bint\s+main\s*\(")


@dataclass
class MainCandidate:
    relpath: str
    score: int
    reasons: List[str]


@dataclass
class ProjectProfile:
    project_root: str
    has_cmake: bool
    cmake_files: List[str]
    sources_cpp: List[str]
    sources_c: List[str]
    headers: List[str]
    kernels_cl: List[str]
    main_candidates: List[MainCandidate]
    selected_entrypoint: Optional[str]
    file_index: List[Dict[str, object]]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["main_candidates"] = [asdict(c) for c in self.main_candidates]
        return d


def _is_text_source(path: str) -> bool:
    return path.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cl"))


def _read_text_best_effort(path: str, max_bytes: int = 1024 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_project(project_root: str) -> ProjectProfile:
    project_root = os.path.abspath(project_root)

    cmake_files: List[str] = []
    sources_cpp: List[str] = []
    sources_c: List[str] = []
    headers: List[str] = []
    kernels_cl: List[str] = []
    file_index: List[Dict[str, object]] = []

    # Walk project, skipping NNPort internal dirs.
    for root, dirs, files in os.walk(project_root):
        # prune internal/run dirs
        dirs[:] = [d for d in dirs if not d.startswith("__pycache__")]
        # Also prune legacy hidden artifact dirs if they exist in a project.
        dirs[:] = [d for d in dirs if d not in (".nnport", ".nnport_backups")]
        # Prune NNPort workspace artifacts (visible policy: build/nnport)
        if os.path.basename(root) in ("build", "Build", "out", "Out"):
            dirs[:] = [d for d in dirs if d != "nnport"]

        for name in files:
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, project_root)

            # Lightweight file index for reproducibility / telemetry.
            try:
                st = os.stat(abs_path)
                rec = {"path": rel, "size": int(st.st_size)}
                # Hash only likely-text sources to avoid wasting time on binaries.
                if _is_text_source(rel):
                    rec["sha256"] = _sha256_file(abs_path)
                file_index.append(rec)
            except Exception:
                pass

            if name == "CMakeLists.txt":
                cmake_files.append(rel)

            if name.endswith((".cc", ".cpp", ".cxx")):
                sources_cpp.append(rel)
            elif name.endswith(".c"):
                sources_c.append(rel)
            elif name.endswith((".h", ".hh", ".hpp")):
                headers.append(rel)
            elif name.endswith(".cl"):
                kernels_cl.append(rel)

    has_cmake = bool(cmake_files)

    # Main candidate scoring
    candidates: List[MainCandidate] = []
    for rel in sources_cpp + sources_c:
        abs_path = os.path.join(project_root, rel)
        text = _read_text_best_effort(abs_path)
        if not text:
            continue
        if not _MAIN_RE.search(text):
            continue

        score = 0
        reasons: List[str] = ["contains main()"]
        base = os.path.basename(rel).lower()

        if base.startswith("main."):
            score += 100
            reasons.append("filename suggests entrypoint")

        # Heuristics that match NNPort conventions
        for token, pts, why in [
            ("input.bin", 10, "reads input.bin"),
            ("output.bin", 10, "writes output.bin"),
            ("kernel.cl", 5, "references kernel.cl"),
            ("clBuildProgram", 3, "uses OpenCL build"),
            ("clEnqueueNDRangeKernel", 3, "launches kernel"),
        ]:
            if token in text:
                score += pts
                reasons.append(why)

        candidates.append(MainCandidate(relpath=rel, score=score, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected = candidates[0].relpath if candidates else None

    return ProjectProfile(
        project_root=project_root,
        has_cmake=has_cmake,
        cmake_files=sorted(cmake_files),
        sources_cpp=sorted(sources_cpp),
        sources_c=sorted(sources_c),
        headers=sorted(headers),
        kernels_cl=sorted(kernels_cl),
        main_candidates=candidates,
        selected_entrypoint=selected,
        file_index=file_index,
    )


def write_profile(profile: ProjectProfile, run_dir: str) -> str:
    """Persist the profile for UI/debugging. Returns the written path."""
    import json

    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    return out_path

