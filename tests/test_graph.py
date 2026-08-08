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

INVALID_PATCH = """diff --git a/../escape.txt b/../escape.txt
--- a/../escape.txt
+++ b/../escape.txt
@@ -0,0 +1 @@
+nope
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


def test_graph_stops_after_second_identical_failure_signature(buggy_repo: Path) -> None:
    worker = SequenceWorker([BAD_PATCH, BAD_PATCH, BAD_PATCH])
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "blocked"
    assert result.attempts == 2
    assert worker.calls == 2
    assert "repeated failure signature" in result.error_log


def test_graph_feeds_patch_validation_error_into_retry(buggy_repo: Path) -> None:
    class FeedbackWorker(SequenceWorker):
        def __init__(self) -> None:
            super().__init__([INVALID_PATCH, GOOD_PATCH])
            self.retry_error_log = ""

        def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
            if attempt == 2:
                self.retry_error_log = error_log
            return super().generate_patch(
                issue=issue, context=context, error_log=error_log, attempt=attempt
            )

    worker = FeedbackWorker()
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "passed"
    assert result.attempts == 2
    assert "PATCH VALIDATION FAILED" in worker.retry_error_log
    assert "unsafe repository path" in worker.retry_error_log
