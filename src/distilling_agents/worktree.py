from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .tools import ToolError, ensure_git_repo


@contextmanager
def temporary_worktree(source_repo: Path) -> Iterator[Path]:
    """Create a detached disposable git worktree and always remove it."""

    source_repo = ensure_git_repo(source_repo)
    root = Path(tempfile.mkdtemp(prefix="distilling-agents-"))
    worktree = root / "worktree"
    add = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=source_repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if add.returncode != 0:
        shutil.rmtree(root, ignore_errors=True)
        raise ToolError(add.stderr.strip() or "failed to create temporary git worktree")
    try:
        yield worktree
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=source_repo,
            text=True,
            capture_output=True,
            check=False,
        )
        shutil.rmtree(root, ignore_errors=True)
