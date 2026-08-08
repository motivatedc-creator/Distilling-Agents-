from __future__ import annotations

from pathlib import Path

import distilling_agents.graph as graph_module
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

BAD_PATCH_2 = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a / b
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
        self.error_logs: list[str] = []

    def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
        self.calls += 1
        self.error_logs.append(error_log)
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


def test_graph_blocks_when_exact_same_patch_is_proposed_twice(buggy_repo: Path) -> None:
    worker = SequenceWorker([BAD_PATCH, BAD_PATCH, GOOD_PATCH])
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
    assert "repeated identical patch" in result.error_log.lower()


def test_same_failure_with_different_patch_logs_stall_and_keeps_trying(
    buggy_repo: Path, monkeypatch
) -> None:
    worker = SequenceWorker([BAD_PATCH, BAD_PATCH_2, GOOD_PATCH])
    real_run_tests = graph_module.run_tests
    calls = 0

    def controlled_tests(repo: Path, command: tuple[str, ...]):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return "fail", "FAILED test_calculator.py::test_add\nE AssertionError: still wrong"
        return real_run_tests(repo, command)

    monkeypatch.setattr(graph_module, "run_tests", controlled_tests)
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "passed"
    assert result.attempts == 3
    assert worker.calls == 3
    assert "stall" in worker.error_logs[2].lower()


def test_graph_retries_worker_exception_instead_of_escaping(buggy_repo: Path) -> None:
    class FlakyWorker(SequenceWorker):
        def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
            self.calls += 1
            self.error_logs.append(error_log)
            if attempt == 1:
                raise RuntimeError("temporary vllm failure")
            return PatchResponse(diff=GOOD_PATCH)

    worker = FlakyWorker([])
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "passed"
    assert result.attempts == 2
    assert "temporary vllm failure" in worker.error_logs[1]


def test_graph_blocks_after_worker_exception_budget(buggy_repo: Path) -> None:
    class BrokenWorker:
        def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
            raise RuntimeError("vllm unavailable")

    result = run_agent(
        worker=BrokenWorker(),
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=2,
    )
    assert result.status == "blocked"
    assert result.attempts == 2
    assert "vllm unavailable" in result.error_log


def test_graph_retries_execution_exception_instead_of_escaping(buggy_repo: Path, monkeypatch) -> None:
    worker = SequenceWorker([GOOD_PATCH, GOOD_PATCH])
    real_run_tests = graph_module.run_tests
    calls = 0

    def flaky_tests(repo: Path, command: tuple[str, ...]):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("test runner crashed")
        return real_run_tests(repo, command)

    monkeypatch.setattr(graph_module, "run_tests", flaky_tests)
    result = run_agent(
        worker=worker,
        issue_description="add(2, 3) must return 5",
        worktree_path=buggy_repo,
        test_command=("python", "-m", "pytest", "-q"),
        max_attempts=3,
    )
    assert result.status == "passed"
    assert result.attempts == 2
    assert "test runner crashed" in worker.error_logs[1]


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
