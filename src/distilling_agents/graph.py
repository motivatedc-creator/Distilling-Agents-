from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .context import build_context
from .models import AgentResult, AgentState
from .tools import (
    apply_patch,
    failure_fingerprint,
    git_diff,
    patch_fingerprint,
    reset_worktree,
    run_tests,
    validate_patch,
)
from .worker import PatchWorker


def _bounded_error(text: str, *, max_chars: int = 6_000) -> str:
    if len(text) <= max_chars:
        return text
    return "<error feedback truncated>\n" + text[-max_chars:]


def build_graph(worker: PatchWorker):
    def retrieve_context(state: AgentState) -> AgentState:
        repo = Path(state["worktree_path"])
        pack = build_context(
            repo,
            issue=state["issue_description"],
            failure_feedback=state.get("error_log", ""),
        )
        return {
            "context": pack.text,
            "context_files": list(pack.files),
            "context_chars": pack.char_count,
            "approximate_context_tokens": pack.approximate_tokens,
        }

    def generate_patch(state: AgentState) -> AgentState:
        attempt = state.get("attempts", 0) + 1
        try:
            response = worker.generate_patch(
                issue=state["issue_description"],
                context=state["context"],
                error_log=state.get("error_log", ""),
                attempt=attempt,
            )
        except Exception as exc:
            return {
                "attempts": attempt,
                "patch": "",
                "current_patch_hash": "",
                "repeated_patch": False,
                "generation_failed": True,
                "execution_failed": False,
                "patch_valid": False,
                "validation_error": "",
                "test_result": "not_run",
                "error_log": _bounded_error(f"WORKER ERROR: {type(exc).__name__}: {exc}"),
                "status": "running",
            }

        current_hash = patch_fingerprint(response.diff)
        previous_failed_hash = state.get("previous_failed_patch_hash", "")
        repeated_patch = bool(previous_failed_hash and current_hash == previous_failed_hash)
        error_log = state.get("error_log", "")
        if repeated_patch:
            error_log = (
                f"REPEATED IDENTICAL PATCH: {current_hash}. "
                "The worker proposed the same previously failed patch again."
            )
        return {
            "attempts": attempt,
            "patch": response.diff,
            "current_patch_hash": current_hash,
            "repeated_patch": repeated_patch,
            "generation_failed": False,
            "execution_failed": False,
            "patch_valid": False,
            "validation_error": "",
            "test_result": "not_run",
            "error_log": _bounded_error(error_log),
            "status": "running",
        }

    def validate_generated_patch(state: AgentState) -> AgentState:
        allowed = set(state.get("context_files", []))
        valid, error = validate_patch(
            Path(state["worktree_path"]),
            state["patch"],
            allowed_paths=allowed,
        )
        update: AgentState = {"patch_valid": valid, "validation_error": error}
        if not valid:
            update["previous_failed_patch_hash"] = state.get("current_patch_hash", "")
            update["error_log"] = _bounded_error(
                f"PATCH VALIDATION FAILED:\n{error}\n\nREJECTED PATCH:\n{state['patch']}"
            )
        return update

    def apply_and_test(state: AgentState) -> AgentState:
        repo = Path(state["worktree_path"])
        allowed = set(state.get("context_files", []))
        try:
            apply_patch(repo, state["patch"], allowed_paths=allowed)
            test_run = run_tests(repo, state["test_command"])
            result, output = test_run
            exit_code = getattr(test_run, "exit_code", 0 if result == "pass" else 1)
        except Exception as exc:
            cleanup_error = ""
            try:
                reset_worktree(repo)
            except Exception as cleanup_exc:
                cleanup_error = f"; cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}"
            return {
                "test_result": "fail",
                "execution_failed": True,
                "error_log": _bounded_error(
                    f"EXECUTION ERROR: {type(exc).__name__}: {exc}{cleanup_error}"
                ),
                "status": "running",
            }

        if result == "pass":
            return {
                "test_result": "pass",
                "execution_failed": False,
                "failure_stall": False,
                "error_log": output,
                "final_diff": git_diff(repo),
                "status": "passed",
            }

        fingerprint = failure_fingerprint(result, output, exit_code)
        previous_fingerprint = state.get("failure_fingerprint", "")
        same_failure = bool(previous_fingerprint and fingerprint == previous_fingerprint)
        failed_diff = git_diff(repo)
        reset_worktree(repo)

        stall_note = ""
        if same_failure:
            stall_note = (
                f"STALL: same failure signature {fingerprint} after a different patch. "
                "Retry remains allowed.\n\n"
            )
        return {
            "test_result": result,
            "execution_failed": False,
            "failure_fingerprint": fingerprint,
            "failure_stall": same_failure,
            "previous_failed_patch_hash": state.get("current_patch_hash", ""),
            "error_log": _bounded_error(
                f"{stall_note}TEST RESULT: {result} (exit={exit_code})\n\n{output}\n\n"
                f"FAILED PATCH:\n{failed_diff[-3_000:]}"
            ),
            "status": "running",
        }

    def after_generate(state: AgentState) -> Literal["validate_patch", "retrieve_context", "blocked"]:
        if state.get("generation_failed", False):
            if state["attempts"] >= state["max_attempts"]:
                return "blocked"
            return "retrieve_context"
        if state.get("repeated_patch", False):
            return "blocked"
        return "validate_patch"

    def after_validation(state: AgentState) -> Literal["apply_and_test", "retrieve_context", "blocked"]:
        if state["patch_valid"]:
            return "apply_and_test"
        if state["attempts"] >= state["max_attempts"]:
            return "blocked"
        return "retrieve_context"

    def after_test(state: AgentState) -> Literal["passed", "retrieve_context", "blocked"]:
        if state["test_result"] == "pass":
            return "passed"
        if state["attempts"] >= state["max_attempts"]:
            return "blocked"
        return "retrieve_context"

    def blocked(state: AgentState) -> AgentState:
        message = state.get("error_log") or state.get("validation_error") or "attempt budget exhausted"
        return {"status": "blocked", "error_log": _bounded_error(message)}

    def passed(state: AgentState) -> AgentState:
        return {"status": "passed"}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_patch", generate_patch)
    graph.add_node("validate_patch", validate_generated_patch)
    graph.add_node("apply_and_test", apply_and_test)
    graph.add_node("blocked", blocked)
    graph.add_node("passed", passed)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_patch")
    graph.add_conditional_edges("generate_patch", after_generate)
    graph.add_conditional_edges("validate_patch", after_validation)
    graph.add_conditional_edges("apply_and_test", after_test)
    graph.add_edge("blocked", END)
    graph.add_edge("passed", END)
    return graph.compile()


def run_agent(
    *,
    worker: PatchWorker,
    issue_description: str,
    worktree_path: Path,
    test_command: tuple[str, ...],
    max_attempts: int = 3,
) -> AgentResult:
    initial_state: AgentState = {
        "issue_description": issue_description,
        "worktree_path": str(worktree_path),
        "test_command": test_command,
        "max_attempts": max_attempts,
        "attempts": 0,
        "context": "",
        "context_files": [],
        "context_chars": 0,
        "approximate_context_tokens": 0,
        "patch": "",
        "current_patch_hash": "",
        "previous_failed_patch_hash": "",
        "repeated_patch": False,
        "patch_valid": False,
        "validation_error": "",
        "generation_failed": False,
        "execution_failed": False,
        "test_result": "not_run",
        "error_log": "",
        "failure_fingerprint": "",
        "failure_stall": False,
        "final_diff": "",
        "status": "running",
    }
    graph = build_graph(worker)
    try:
        final = graph.invoke(initial_state, {"recursion_limit": 30})
    except Exception as exc:
        return AgentResult(
            status="blocked",
            attempts=initial_state["attempts"],
            error_log=_bounded_error(f"FATAL HARNESS ERROR: {type(exc).__name__}: {exc}"),
        )
    return AgentResult(
        status=final["status"],
        attempts=final["attempts"],
        diff=final.get("final_diff", ""),
        error_log=final.get("error_log", ""),
    )
