from __future__ import annotations

from pathlib import Path

from distilling_agents.graph import run_agent
from distilling_agents.models import PatchResponse


BAD_PATCH = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a * b
"""

GOOD_PATCH = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


class SequenceWorker:
    def __init__(self, patches: list[str]) -> None:
        self.patches = iter(patches)
        self.calls = 0

    def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
        self.calls += 1
        return PatchResponse(diff=next(self.patches))


def test_graph_retries_failed_patch_and_returns_passing_diff(buggy_repo: Path) -> None:
    worker = SequenceWorker([BAD_PATCH, GOOD_PATCH])
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "passed"
    assert result.attempts == 2
    assert "+    return a + b" in result.diff
    assert worker.calls == 2


def test_graph_stops_after_attempt_budget(buggy_repo: Path) -> None:
    worker = SequenceWorker([BAD_PATCH, BAD_PATCH, BAD_PATCH])
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "blocked"
    assert result.attempts == 3
    assert worker.calls == 3
