"""Tests for case-insensitive path resolution.

Prevents subagents from creating duplicate directories that differ only
in case (e.g. NeonDrift vs neondrift) when git clone, cd, mkdir, or
file_ops resolve paths.
"""

import pytest
from pathlib import Path

from agent.tools.bash import BashTool, resolve_case_insensitive


class TestResolveCaseInsensitive:
    """The utility must find existing dirs regardless of case."""

    def test_exact_match_returns_same(self, tmp_path):
        (tmp_path / "myproject").mkdir()
        assert resolve_case_insensitive(tmp_path, "myproject") == "myproject"

    def test_case_mismatch_returns_existing(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        assert resolve_case_insensitive(tmp_path, "NeonDrift") == "neondrift"

    def test_no_match_returns_original(self, tmp_path):
        assert resolve_case_insensitive(tmp_path, "brand-new") == "brand-new"

    def test_files_ignored(self, tmp_path):
        (tmp_path / "README.md").touch()
        # Files shouldn't match — only directories
        assert resolve_case_insensitive(tmp_path, "readme.md") == "readme.md"

    def test_multiple_dirs_picks_match(self, tmp_path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        assert resolve_case_insensitive(tmp_path, "ALPHA") == "alpha"
        assert resolve_case_insensitive(tmp_path, "BETA") == "beta"


class TestBashToolCaseFix:
    """BashTool._fix_case_sensitive_paths must rewrite commands."""

    def _make_tool(self, tmp_path):
        return BashTool(workspace=tmp_path, allow_git=True)

    def test_git_clone_no_target_rewritten(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        tool = self._make_tool(tmp_path)
        cmd = "git clone git@github.com:user/NeonDrift.git"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert "neondrift" in fixed
        assert "NeonDrift.git" in fixed  # URL unchanged

    def test_git_clone_explicit_target_rewritten(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        tool = self._make_tool(tmp_path)
        cmd = "git clone git@github.com:user/NeonDrift.git NeonDrift"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert fixed.endswith("neondrift") or "neondrift" in fixed.split()[-1]

    def test_git_clone_new_repo_unchanged(self, tmp_path):
        tool = self._make_tool(tmp_path)
        cmd = "git clone git@github.com:user/NewRepo.git"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert fixed == cmd

    def test_cd_rewritten(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        tool = self._make_tool(tmp_path)
        cmd = "cd NeonDrift && npm install"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert "neondrift" in fixed
        assert "npm install" in fixed

    def test_cd_new_dir_unchanged(self, tmp_path):
        tool = self._make_tool(tmp_path)
        cmd = "cd newdir && ls"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert fixed == cmd

    def test_mkdir_rewritten(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        tool = self._make_tool(tmp_path)
        cmd = "mkdir NeonDrift"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert "neondrift" in fixed

    def test_mkdir_p_rewritten(self, tmp_path):
        (tmp_path / "neondrift").mkdir()
        tool = self._make_tool(tmp_path)
        cmd = "mkdir -p NeonDrift/src"
        fixed = tool._fix_case_sensitive_paths(cmd)
        assert "neondrift" in fixed

    def test_unrelated_command_unchanged(self, tmp_path):
        tool = self._make_tool(tmp_path)
        cmd = "echo hello world"
        assert tool._fix_case_sensitive_paths(cmd) == cmd


class TestFileOpsCaseInsensitive:
    """FileOpsTool must resolve paths case-insensitively."""

    def test_resolve_existing_subdir(self, tmp_path):
        from agent.tools.file_ops import FileOpsTool
        (tmp_path / "neondrift" / "src").mkdir(parents=True)

        tool = FileOpsTool(workspace=tmp_path)
        resolved = tool._resolve_safe("NeonDrift/src/main.ts", for_write=True)
        assert resolved is not None
        # Should resolve through the existing "neondrift" dir
        assert "neondrift" in str(resolved)
        assert "NeonDrift" not in str(resolved)

    def test_resolve_new_path_unchanged(self, tmp_path):
        from agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=tmp_path)
        resolved = tool._resolve_safe("newproject/src/main.ts", for_write=True)
        assert resolved is not None
        assert "newproject" in str(resolved)
