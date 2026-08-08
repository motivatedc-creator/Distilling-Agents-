from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from .graph import run_agent
from .worker import VLLMWorker
from .worktree import temporary_worktree


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the constrained local code-repair loop.")
    p.add_argument("repo", type=Path, help="Path to a git repository to repair.")
    p.add_argument("--issue", required=True, help="Bug/issue description.")
    p.add_argument("--test-command", default="python -m pytest -q", help="Deterministic test command.")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"))
    return p


def main() -> int:
    args = parser().parse_args()
    worker = VLLMWorker(base_url=args.base_url, model=args.model)
    with temporary_worktree(args.repo) as worktree:
        result = run_agent(
            worker=worker,
            issue_description=args.issue,
            worktree_path=worktree,
            test_command=tuple(shlex.split(args.test_command)),
            max_attempts=args.max_attempts,
        )
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
