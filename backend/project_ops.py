"""Project file operations (LLM JSON ops) validation and application helpers.

This module is intentionally strict: we treat all LLM-proposed edits as untrusted input.
"""

from __future__ import annotations

import hashlib
import os
import json
import shutil
import time
from typing import Any, Dict, List, Optional, Set


class OpsValidationError(Exception):
    pass


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
        return {"backup_root": backup_root, "applied_ops": normalized, "created": sorted(created)}

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


