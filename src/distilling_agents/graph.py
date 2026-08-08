from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .context import build_context
from .models import AgentResult, AgentState
from .tools import apply_patch, git_diff, reset_worktree, run_tests, validate_patch
from .worker import PatchWorker


def build_graph(worker: PatchWorker):
    def retrieve_context(state: AgentState) -> AgentState:
        repo = Path(state["worktree_path"])
        return {"context": build_context(repo)}

    def generate_patch(state: AgentState) -> AgentState:
        attempt = state.get("attempts", 0) + 1
        response = worker.generate_patch(
            issue=state["issue_description"],
            context=state["context"],
            error_log=state.get("error_log", ""),
            attempt=attempt,
        )
        return {
            "attempts": attempt,
            "patch": response.diff,
            "patch_valid": False,
            "validation_error": "",
            "test_result": "not_run",
            "status": "running",
        }

    def validate_generated_patch(state: AgentState) -> AgentState:
        valid, error = validate_patch(Path(state["worktree_path"]), state["patch"])
        update: AgentState = {"patch_valid": valid, "validation_error": error}
        if not valid:
            update["error_log"] = (
                f"PATCH VALIDATION FAILED:\n{error}\n\nREJECTED PATCH:\n{state['patch']}"
            )
        return update

    def apply_and_test(state: AgentState) -> AgentState:
        repo = Path(state["worktree_path"])
        apply_patch(repo, state["patch"])
        result, output = run_tests(repo, state["test_command"])
        if result == "pass":
            return {
                "test_result": "pass",
                "error_log": output,
                "final_diff": git_diff(repo),
                "status": "passed",
            }

        failed_diff = git_diff(repo)
        reset_worktree(repo)
        return {
            "test_result": result,
            "error_log": f"TEST RESULT: {result}\n\n{output}\n\nFAILED PATCH:\n{failed_diff}",
            "status": "running",
        }

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
        message = state.get("validation_error") or state.get("error_log", "") or "attempt budget exhausted"
        return {"status": "blocked", "error_log": message}

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
    graph.add_edge("generate_patch", "validate_patch")
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
    graph = build_graph(worker)
    final = graph.invoke(
        {
            "issue_description": issue_description,
            "worktree_path": str(worktree_path),
            "test_command": test_command,
            "max_attempts": max_attempts,
            "attempts": 0,
            "patch": "",
            "patch_valid": False,
            "validation_error": "",
            "test_result": "not_run",
            "error_log": "",
            "final_diff": "",
            "status": "running",
        },
        {"recursion_limit": 30},
    )
    return AgentResult(
        status=final["status"],
        attempts=final["attempts"],
        diff=final.get("final_diff", ""),
        error_log=final.get("error_log", ""),
    )
