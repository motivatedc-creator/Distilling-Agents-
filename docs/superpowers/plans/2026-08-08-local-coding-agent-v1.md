# Local Coding Agent V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimum viable local coding-agent loop that safely turns an issue description into a tested unified diff using Qwen2.5-Coder behind vLLM.

**Architecture:** The LLM is a patch generator, not a shell operator. LangGraph controls context retrieval, patch generation, validation, deterministic test execution, retries, and termination. Every CLI run operates in a detached temporary Git worktree so the source checkout remains untouched.

**Tech Stack:** Python 3.11+, LangGraph, OpenAI Python client against vLLM's OpenAI-compatible API, Pydantic v2, Git, pytest.

## Global Constraints

- One GPU model slot in V1: Qwen2.5-Coder-7B-Instruct-AWQ.
- Maximum worker attempts: 3 by default.
- No arbitrary shell tool exposed to the model.
- No orchestrator, marketing engine, Docker sandbox, or PR automation in V1.
- Invalid patches must be rejected before application.
- The source repository checkout must not be modified.

---

### Task 1: Define the worker contract and deterministic repository tools

**Files:**
- Create: `src/distilling_agents/models.py`
- Create: `src/distilling_agents/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `PatchResponse`, `validate_patch()`, `apply_patch()`, `run_tests()`, `git_diff()`, `reset_worktree()`.

- [x] Write tests for safe patch validation and test execution.
- [x] Implement repository-root path validation.
- [x] Validate with `git apply --check` before application.
- [x] Run tests through one harness-owned command tuple.
- [x] Verify the tests pass.

### Task 2: Add the Qwen/vLLM worker client

**Files:**
- Create: `src/distilling_agents/worker.py`

**Interfaces:**
- Produces: `PatchWorker` protocol and `VLLMWorker.generate_patch(...) -> PatchResponse`.

- [x] Use vLLM's OpenAI-compatible endpoint.
- [x] Request `json_schema` structured output using the Pydantic schema.
- [x] Keep the contract limited to a single unified diff.

### Task 3: Implement the bounded LangGraph repair loop

**Files:**
- Create: `src/distilling_agents/context.py`
- Create: `src/distilling_agents/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces: `build_graph(worker)` and `run_agent(...) -> AgentResult`.

- [x] Retrieve a bounded text context pack.
- [x] Generate and validate patches before application.
- [x] Run tests and return a final diff on success.
- [x] Reset failed patches before retrying.
- [x] Stop at the attempt budget with `blocked` status.
- [x] Verify a synthetic bug succeeds on retry and repeated failure terminates.

### Task 4: Isolate execution and expose a CLI

**Files:**
- Create: `src/distilling_agents/worktree.py`
- Create: `src/distilling_agents/cli.py`

**Interfaces:**
- Produces: `temporary_worktree(source_repo)` and `distill-agent` CLI.

- [x] Create a detached disposable Git worktree.
- [x] Always remove it after the run.
- [x] Accept issue text, deterministic test command, model endpoint, and attempt budget.
- [x] Return JSON and a non-zero exit status when blocked.

### Task 5: Package, document, and continuously verify

**Files:**
- Create: `pyproject.toml`
- Create: `.github/workflows/ci.yml`
- Create: `README.md`

- [x] Package for Python 3.11+.
- [x] Add pytest dev dependency and console entry point.
- [x] Document vLLM startup and the V1 safety boundary.
- [x] Add GitHub Actions CI on Python 3.11.
- [ ] Next benchmark gate: create 10 synthetic bug fixtures and require >=8 autonomous repairs before adding an orchestrator.
