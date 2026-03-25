"""File operations tool — read, write, list, edit files in workspace.

All paths are restricted to the session workspace. No access to
system files, dotfiles, or anything outside the workspace root.
"""

import os
from pathlib import Path

import structlog

log = structlog.get_logger()

MAX_FILE_SIZE = 1_000_000  # 1MB max read/write
MAX_FILES_LIST = 200


class FileOpsTool:
    """File operations scoped to a workspace directory."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve_safe(self, path: str) -> Path | None:
        """Resolve a path within the workspace. Returns None if it escapes."""
        try:
            resolved = (self.workspace / path).resolve()
            # Must be inside workspace
            if not str(resolved).startswith(str(self.workspace)):
                return None
            return resolved
        except (ValueError, OSError):
            return None

    async def read(self, path: str) -> dict:
        """Read a file's contents."""
        resolved = self._resolve_safe(path)
        if not resolved:
            return {"error": f"Path '{path}' is outside workspace"}

        if not resolved.exists():
            return {"error": f"File not found: {path}"}

        if not resolved.is_file():
            return {"error": f"Not a file: {path}"}

        size = resolved.stat().st_size
        if size > MAX_FILE_SIZE:
            return {"error": f"File too large: {size} bytes (max {MAX_FILE_SIZE})"}

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            log.debug("file_read", path=path, size=len(content))
            return {"content": content, "path": path, "size": len(content)}
        except Exception as e:
            return {"error": f"Read failed: {e}"}

    async def write(self, path: str, content: str) -> dict:
        """Write content to a file (creates parent dirs as needed)."""
        resolved = self._resolve_safe(path)
        if not resolved:
            return {"error": f"Path '{path}' is outside workspace"}

        if len(content) > MAX_FILE_SIZE:
            return {"error": f"Content too large: {len(content)} bytes (max {MAX_FILE_SIZE})"}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            log.info("file_written", path=path, size=len(content))
            return {"path": path, "size": len(content), "created": True}
        except Exception as e:
            return {"error": f"Write failed: {e}"}

    async def edit(self, path: str, old: str, new: str) -> dict:
        """Replace a string in a file (first occurrence only)."""
        resolved = self._resolve_safe(path)
        if not resolved:
            return {"error": f"Path '{path}' is outside workspace"}

        if not resolved.exists():
            return {"error": f"File not found: {path}"}

        try:
            content = resolved.read_text(encoding="utf-8")
            if old not in content:
                return {"error": f"String not found in {path}"}

            new_content = content.replace(old, new, 1)
            resolved.write_text(new_content, encoding="utf-8")
            log.info("file_edited", path=path)
            return {"path": path, "replaced": True}
        except Exception as e:
            return {"error": f"Edit failed: {e}"}

    async def list_files(self, path: str = ".") -> dict:
        """List files and directories in a path."""
        resolved = self._resolve_safe(path)
        if not resolved:
            return {"error": f"Path '{path}' is outside workspace"}

        if not resolved.exists():
            return {"error": f"Directory not found: {path}"}

        if not resolved.is_dir():
            return {"error": f"Not a directory: {path}"}

        try:
            entries = []
            for i, entry in enumerate(sorted(resolved.iterdir())):
                if i >= MAX_FILES_LIST:
                    entries.append(f"... ({MAX_FILES_LIST}+ entries, truncated)")
                    break

                rel = entry.relative_to(self.workspace)
                if entry.is_dir():
                    entries.append(f"{rel}/")
                else:
                    size = entry.stat().st_size
                    entries.append(f"{rel} ({size} bytes)")

            return {"path": path, "entries": entries, "count": len(entries)}
        except Exception as e:
            return {"error": f"List failed: {e}"}

    async def delete(self, path: str) -> dict:
        """Delete a file (not directories)."""
        resolved = self._resolve_safe(path)
        if not resolved:
            return {"error": f"Path '{path}' is outside workspace"}

        if not resolved.exists():
            return {"error": f"File not found: {path}"}

        if resolved.is_dir():
            return {"error": "Cannot delete directories (safety restriction)"}

        try:
            resolved.unlink()
            log.info("file_deleted", path=path)
            return {"path": path, "deleted": True}
        except Exception as e:
            return {"error": f"Delete failed: {e}"}