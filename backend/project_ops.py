"""Project file operations (LLM JSON ops) validation and application helpers.

This module is intentionally strict: we treat all LLM-proposed edits as untrusted input.
"""

from __future__ import annotations

import hashlib
import os
import json
import shutil
import time
import difflib
from typing import Any, Dict, List, Optional, Set


class OpsValidationError(Exception):
    pass


_PROTECTED_BEGIN = "NNPORT_PROTECTED_BEGIN"
_PROTECTED_END = "NNPORT_PROTECTED_END"


def _extract_protected_blocks(text: str) -> List[str]:
    """Return a list of protected blocks (inclusive) found in text.

    A protected block is any region between lines containing NNPORT_PROTECTED_BEGIN
    and NNPORT_PROTECTED_END. The content inside must remain byte-for-byte identical
    across updates, otherwise the op is rejected.
    """
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        if _PROTECTED_BEGIN in lines[i]:
            start = i
            j = i + 1
            while j < len(lines) and _PROTECTED_END not in lines[j]:
                j += 1
            if j < len(lines):
                end = j
                blocks.append("".join(lines[start : end + 1]))
                i = end + 1
                continue
        i += 1
    return blocks


def _enforce_protected_blocks(old_text: str, new_text: str, *, path: str) -> None:
    """Reject updates that modify or remove protected blocks."""
    old_blocks = _extract_protected_blocks(old_text)
    if not old_blocks:
        return
    new_blocks = _extract_protected_blocks(new_text)
    if old_blocks != new_blocks:
        raise OpsValidationError(
            f"Protected region modified/removed in {path}. "
            f"Do not edit text between {_PROTECTED_BEGIN} and {_PROTECTED_END}."
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _is_safe_relpath(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    if "\x00" in path:
        return False
    # No absolute paths (posix or windows-style)
    if os.path.isabs(path):
        return False
    # Disallow drive letters (basic check)
    if len(path) >= 2 and path[1] == ":":
        return False
    norm = os.path.normpath(path)
    # Disallow traversal
    if norm == ".." or norm.startswith(".." + os.sep):
        return False
    # Disallow sneaky current-dir absolute-ish
    if norm.startswith(os.sep):
        return False
    return True


def _safe_abspath(project_root: str, relpath: str) -> str:
    if not _is_safe_relpath(relpath):
        raise OpsValidationError(f"Unsafe path: {relpath!r}")
    root_real = os.path.realpath(project_root)
    abs_candidate = os.path.realpath(os.path.join(project_root, relpath))
    if abs_candidate == root_real or abs_candidate.startswith(root_real + os.sep):
        return abs_candidate
    raise OpsValidationError(f"Path escapes project root: {relpath!r}")


def validate_ops_object(
    obj: Any,
    project_root: str,
    *,
    allow_delete: bool = True,
    allow_rename: bool = True,
    require_expected_sha_for_update: bool = False,
) -> List[Dict[str, Any]]:
    """Validate and normalize an LLM-proposed ops object.

    Returns a normalized list of ops (dicts).
    """

    if not isinstance(obj, dict):
        raise OpsValidationError("Top-level JSON must be an object.")

    ops = obj.get("ops")
    if not isinstance(ops, list) or not ops:
        raise OpsValidationError('Top-level JSON must contain non-empty key "ops": [...].')

    if not isinstance(project_root, str) or not project_root:
        raise OpsValidationError("project_root must be a non-empty string")
    if not os.path.isdir(project_root):
        raise OpsValidationError(f"project_root does not exist or is not a directory: {project_root}")

    normalized: List[Dict[str, Any]] = []

    for idx, op in enumerate(ops):
        if not isinstance(op, dict):
            raise OpsValidationError(f"ops[{idx}] must be an object")

        op_type = op.get("type")
        if op_type not in ("create", "update", "delete", "rename"):
            raise OpsValidationError(
                f"ops[{idx}].type must be one of create/update/delete/rename (got {op_type!r})"
            )

        expected_before = op.get("expected_sha256_before")
        if expected_before is not None:
            if not isinstance(expected_before, str) or len(expected_before) != 64:
                raise OpsValidationError(
                    f"ops[{idx}].expected_sha256_before must be a 64-char hex sha256 string"
                )

        if op_type in ("create", "update", "delete"):
            path = op.get("path")
            if not isinstance(path, str) or not path:
                raise OpsValidationError(f"ops[{idx}].path must be a non-empty string")
            _ = _safe_abspath(project_root, path)

            if op_type in ("create", "update"):
                content = op.get("content")
                if not isinstance(content, str):
                    raise OpsValidationError(f"ops[{idx}].content must be a string for {op_type}")

                if op_type == "update" and require_expected_sha_for_update and not expected_before:
                    raise OpsValidationError(
                        f"ops[{idx}].expected_sha256_before is required for update operations"
                    )

                normalized.append(
                    {
                        "type": op_type,
                        "path": path,
                        "content": content,
                        "expected_sha256_before": expected_before,
                    }
                )
                continue

            # delete
            if not allow_delete:
                raise OpsValidationError("delete operations are disabled")
            normalized.append(
                {
                    "type": "delete",
                    "path": path,
                    "expected_sha256_before": expected_before,
                }
            )
            continue

        # rename
        if not allow_rename:
            raise OpsValidationError("rename operations are disabled")
        from_path = op.get("from_path")
        to_path = op.get("to_path")
        if not isinstance(from_path, str) or not from_path:
            raise OpsValidationError(f"ops[{idx}].from_path must be a non-empty string")
        if not isinstance(to_path, str) or not to_path:
            raise OpsValidationError(f"ops[{idx}].to_path must be a non-empty string")
        _ = _safe_abspath(project_root, from_path)
        _ = _safe_abspath(project_root, to_path)

        normalized.append(
            {
                "type": "rename",
                "from_path": from_path,
                "to_path": to_path,
                "expected_sha256_before": expected_before,
            }
        )

    return normalized


def read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_file_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def compute_existing_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def _backup_file(project_root: str, backup_root: str, relpath: str) -> None:
    src_abs = _safe_abspath(project_root, relpath)
    if not os.path.exists(src_abs):
        return
    dst_abs = os.path.join(backup_root, "files", relpath)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    shutil.copy2(src_abs, dst_abs)


def rollback_from_backup(project_root: str, backup_root: str) -> None:
    """Restore files from a backup created by apply_ops_in_place()."""
    manifest_path = os.path.join(backup_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise OpsValidationError(f"Backup manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    created: List[str] = manifest.get("created", []) or []
    backed_up: List[str] = manifest.get("backed_up", []) or []

    # Delete files we created during the run (best-effort).
    for relpath in created:
        try:
            abs_path = _safe_abspath(project_root, relpath)
            if os.path.exists(abs_path):
                os.remove(abs_path)
        except Exception:
            pass

    # Restore backups.
    for relpath in backed_up:
        src = os.path.join(backup_root, "files", relpath)
        if not os.path.exists(src):
            continue
        dst = _safe_abspath(project_root, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def apply_ops_in_place(
    project_root: str,
    ops_obj: Any,
    *,
    run_id: str,
    backups_root: Optional[str] = None,
    require_expected_sha_for_update: bool = False,
) -> Dict[str, Any]:
    """Validate and apply JSON ops to project_root, creating a rollback backup.

    Backup layout:
      <project_root>/.nnport_backups/<run_id>/
        manifest.json
        files/<relpath>   (copies of pre-edit files)
    """
    normalized = validate_ops_object(
        ops_obj,
        project_root,
        allow_delete=True,
        allow_rename=True,
        require_expected_sha_for_update=require_expected_sha_for_update,
    )

    # Backups location:
    # - if backups_root is provided (preferred), use it
    # - otherwise default to a conventional project-local folder
    backup_base = backups_root or os.path.join(project_root, "build", "nnport", "backups")
    backup_root = os.path.join(backup_base, run_id)
    os.makedirs(backup_root, exist_ok=True)

    backed_up: Set[str] = set()
    created: Set[str] = set()

    # Pre-backup anything that might be modified or removed.
    for op in normalized:
        if op["type"] in ("update", "delete"):
            backed_up.add(op["path"])
        elif op["type"] == "rename":
            backed_up.add(op["from_path"])
            backed_up.add(op["to_path"])  # destination might exist
        elif op["type"] == "create":
            # If it already exists, treat it as a modification and back it up.
            abs_path = _safe_abspath(project_root, op["path"])
            if os.path.exists(abs_path):
                backed_up.add(op["path"])

    for relpath in sorted(backed_up):
        _backup_file(project_root, backup_root, relpath)

    manifest = {
        "run_id": run_id,
        "timestamp": int(time.time()),
        "backed_up": sorted(backed_up),
        "created": [],
        "ops": normalized,
    }
    with open(os.path.join(backup_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Apply ops. If anything fails, rollback.
    try:
        for op in normalized:
            op_type = op["type"]
            expected_before = op.get("expected_sha256_before")

            if op_type in ("update", "delete", "rename"):
                # Check precondition if provided.
                if op_type in ("update", "delete"):
                    abs_path = _safe_abspath(project_root, op["path"])
                    current = compute_existing_sha256(abs_path)
                else:
                    abs_path = _safe_abspath(project_root, op["from_path"])
                    current = compute_existing_sha256(abs_path)
                if expected_before and current and expected_before != current:
                    raise OpsValidationError(
                        f"SHA mismatch for {op_type}: expected {expected_before}, got {current}"
                    )

            if op_type == "create":
                abs_path = _safe_abspath(project_root, op["path"])
                if not os.path.exists(abs_path):
                    created.add(op["path"])
                write_file_text(abs_path, op["content"])
            elif op_type == "update":
                abs_path = _safe_abspath(project_root, op["path"])
                if not os.path.exists(abs_path):
                    raise OpsValidationError(f"Cannot update missing file: {op['path']}")
                # Prevent the model from deleting or rewriting protected regions.
                try:
                    old_text = read_file_text(abs_path)
                except Exception:
                    old_text = ""
                _enforce_protected_blocks(old_text, op["content"], path=op["path"])
                write_file_text(abs_path, op["content"])
            elif op_type == "delete":
                abs_path = _safe_abspath(project_root, op["path"])
                if os.path.exists(abs_path):
                    os.remove(abs_path)
            elif op_type == "rename":
                from_abs = _safe_abspath(project_root, op["from_path"])
                to_abs = _safe_abspath(project_root, op["to_path"])
                if not os.path.exists(from_abs):
                    raise OpsValidationError(f"Cannot rename missing file: {op['from_path']}")
                os.makedirs(os.path.dirname(to_abs), exist_ok=True)
                os.replace(from_abs, to_abs)
            else:
                raise OpsValidationError(f"Unsupported op type: {op_type}")

        # Update manifest with created list after successful apply.
        manifest["created"] = sorted(created)
        with open(os.path.join(backup_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Build lightweight diff previews for UI/logging (best-effort).
        # We diff against the pre-edit backup stored in backup_root/files/<path> when present.
        diffs: Dict[str, str] = {}
        summaries: Dict[str, str] = {}
        changed_files: List[str] = []
        for op in normalized:
            if op["type"] not in ("create", "update"):
                continue
            relpath = op["path"]
            changed_files.append(relpath)
            before_path = os.path.join(backup_root, "files", relpath)
            try:
                before_text = read_file_text(before_path) if os.path.exists(before_path) else ""
            except Exception:
                before_text = ""
            after_text = op.get("content") or ""
            try:
                before_lines = before_text.splitlines()
                after_lines = after_text.splitlines()
                ud = difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"a/{relpath}",
                    tofile=f"b/{relpath}",
                    lineterm="",
                    n=3,
                )
                diff_text = "\n".join(list(ud))
                if diff_text:
                    # Cap diff size to keep websocket payloads sane.
                    diff_lines = diff_text.splitlines()
                    if len(diff_lines) > 200:
                        diff_text = "\n".join(diff_lines[:200]) + "\n... (diff truncated) ..."
                    diffs[relpath] = diff_text
                    # Basic human-ish summary: show first removed line vs first added line.
                    try:
                        removed = None
                        added = None
                        for ln in diff_lines:
                            if ln.startswith("---") or ln.startswith("+++") or ln.startswith("@@"):
                                continue
                            if removed is None and ln.startswith("-"):
                                removed = ln[1:].strip()
                                continue
                            if added is None and ln.startswith("+"):
                                added = ln[1:].strip()
                                continue
                            if removed is not None and added is not None:
                                break
                        if removed and added and removed != added:
                            summaries[relpath] = f"Changed `{removed}` → `{added}`"
                        elif added and not removed:
                            summaries[relpath] = f"Added `{added}`"
                        elif removed and not added:
                            summaries[relpath] = f"Removed `{removed}`"
                    except Exception:
                        pass
            except Exception:
                # non-fatal
                pass

        return {
            "backup_root": backup_root,
            "applied_ops": normalized,
            "created": sorted(created),
            "changed_files": changed_files,
            "diffs": diffs,
            "summaries": summaries,
        }

    except Exception:
        # Best-effort rollback
        try:
            # persist created list for rollback cleanup
            manifest["created"] = sorted(created)
            with open(os.path.join(backup_root, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
        try:
            rollback_from_backup(project_root, backup_root)
        except Exception:
            pass
        raise


