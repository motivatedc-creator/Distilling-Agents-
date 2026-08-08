from __future__ import annotations

from pathlib import Path

from distilling_agents.context import build_context
from distilling_agents.tools import validate_patch


def test_context_prefers_issue_relevant_files_and_caps_size(buggy_repo: Path) -> None:
    unrelated = buggy_repo / "unrelated.py"
    unrelated.write_text("VALUE = 1\n", encoding="utf-8")
    # Track it so retrieval has a real choice to make.
    import subprocess
    subprocess.run(["git", "add", "unrelated.py"], cwd=buggy_repo, check=True)
    subprocess.run(["git", "commit", "-m", "add unrelated"], cwd=buggy_repo, check=True, capture_output=True)

    context = build_context(
        buggy_repo,
        issue="calculator add returns the wrong result",
        max_files=2,
        max_total_chars=2_000,
    )
    assert "### FILE: calculator.py" in context.text
    assert "calculator.py" in context.files
    assert len(context.text) <= 2_000


def test_validate_patch_rejects_file_outside_allowed_targets(buggy_repo: Path) -> None:
    patch = """diff --git a/test_calculator.py b/test_calculator.py
--- a/test_calculator.py
+++ b/test_calculator.py
@@ -2,4 +2,4 @@ from calculator import add
 
 def test_add():
-    assert add(2, 3) == 5
+    assert add(2, 3) == -1
"""
    valid, error = validate_patch(buggy_repo, patch, allowed_paths={"calculator.py"})
    assert valid is False
    assert "outside allowed targets" in error
