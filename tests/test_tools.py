from __future__ import annotations

from pathlib import Path

from distilling_agents.tools import apply_patch, git_diff, run_tests, validate_patch


GOOD_PATCH = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def test_validate_patch_accepts_repo_relative_unified_diff(buggy_repo: Path) -> None:
    valid, error = validate_patch(buggy_repo, GOOD_PATCH)
    assert valid is True
    assert error == ""


def test_validate_patch_rejects_parent_traversal(buggy_repo: Path) -> None:
    patch = """diff --git a/../escape.txt b/../escape.txt
--- a/../escape.txt
+++ b/../escape.txt
@@ -0,0 +1 @@
+nope
"""
    valid, error = validate_patch(buggy_repo, patch)
    assert valid is False
    assert "unsafe repository path" in error


def test_apply_patch_then_tests_pass(buggy_repo: Path) -> None:
    apply_patch(buggy_repo, GOOD_PATCH)
    result, output = run_tests(buggy_repo, ("python", "-m", "pytest", "-q"))
    assert result == "pass", output
    assert "return a + b" in git_diff(buggy_repo)
