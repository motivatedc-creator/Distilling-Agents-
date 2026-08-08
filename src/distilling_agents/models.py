from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class PatchResponse(BaseModel):
    """The only payload the worker model is allowed to return."""

    model_config = ConfigDict(extra="forbid")
    diff: str = Field(min_length=1, description="A unified git diff rooted at the repository.")


class AgentState(TypedDict, total=False):
    issue_description: str
    worktree_path: str
    test_command: tuple[str, ...]
    max_attempts: int
    attempts: int
    context: str
    patch: str
    patch_valid: bool
    validation_error: str
    test_result: Literal["not_run", "pass", "fail", "timeout"]
    error_log: str
    final_diff: str
    status: Literal["running", "passed", "blocked"]


class AgentResult(BaseModel):
    status: Literal["passed", "blocked"]
    attempts: int
    diff: str = ""
    error_log: str = ""
