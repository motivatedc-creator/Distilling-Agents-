from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestRun:
    status: str
    output: str
    exit_code: int | None

    def __iter__(self) -> Iterator[str]:
        # Preserve the existing two-value unpacking API.
        yield self.status
        yield self.output


def _run(
    args: Iterable[str],
    *,
    cwd: Path,
    timeout: int = 30,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {timeout}s") from exc


def ensure_git_repo(repo: Path) -> Path:
    repo = repo.resolve()
    result = _run(["git", "rev-parse", "--show-toplevel"], cwd=repo)
    if result.returncode != 0:
        raise ToolError(f"not a git repository: {repo}")
    return Path(result.stdout.strip()).resolve()


def list_files(repo: Path) -> list[str]:
    repo = ensure_git_repo(repo)
    result = _run(["git", "ls-files", "-z"], cwd=repo)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "git ls-files failed")
    return [item for item in result.stdout.split("\0") if item]


def _safe_repo_path(repo: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or ".." in posix.parts or ".git" in posix.parts:
        raise ToolError(f"unsafe repository path: {relative_path}")
    candidate = (repo / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise ToolError(f"path escapes repository: {relative_path}") from exc
    return candidate


def read_file(repo: Path, relative_path: str, *, max_chars: int = 12_000) -> str:
    repo = ensure_git_repo(repo)
    path = _safe_repo_path(repo, relative_path)
    if not path.is_file():
        raise ToolError(f"file not found: {relative_path}")
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        return data[:max_chars] + "\n...<truncated>...\n"
    return data


def search_repo(repo: Path, pattern: str, *, max_results: int = 50) -> str:
    repo = ensure_git_repo(repo)
    if not pattern.strip():
        return ""
    result = _run(["git", "grep", "-n", "--", pattern], cwd=repo)
    if result.returncode not in (0, 1):
        raise ToolError(result.stderr.strip() or "git grep failed")
    return "\n".join(result.stdout.splitlines()[:max_results])


def _paths_from_diff(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        paths.append(raw)
    return paths


def validate_patch(
    repo: Path,
    diff_text: str,
    *,
    allowed_paths: set[str] | None = None,
) -> tuple[bool, str]:
    repo = ensure_git_repo(repo)
    if not diff_text.strip():
        return False, "empty diff"

    paths = _paths_from_diff(diff_text)
    if not paths:
        return False, "diff contains no file paths"

    try:
        for path in paths:
            _safe_repo_path(repo, path)
    except ToolError as exc:
        return False, str(exc)

    if allowed_paths is not None:
        unexpected = sorted(set(paths) - set(allowed_paths))
        if unexpected:
            return False, f"patch touches files outside allowed targets: {', '.join(unexpected)}"

    result = _run(["git", "apply", "--check", "--whitespace=error", "-"], cwd=repo, input_text=diff_text)
    if result.returncode != 0:
        return False, result.stderr.strip() or "git apply --check rejected patch"
    return True, ""


def apply_patch(
    repo: Path,
    diff_text: str,
    *,
    allowed_paths: set[str] | None = None,
) -> None:
    repo = ensure_git_repo(repo)
    valid, error = validate_patch(repo, diff_text, allowed_paths=allowed_paths)
    if not valid:
        raise ToolError(error)
    result = _run(["git", "apply", "--whitespace=error", "-"], cwd=repo, input_text=diff_text)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "git apply failed")


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    marker = "\n<test output truncated>\n"
    keep = max(0, max_chars - len(marker))
    return marker + output[-keep:]


def run_tests(
    repo: Path,
    command: tuple[str, ...],
    *,
    timeout: int = 60,
    max_output_chars: int = 6_000,
) -> TestRun:
    repo = ensure_git_repo(repo)
    if not command:
        raise ToolError("test command is empty")
    if max_output_chars < 256:
        raise ValueError("max_output_chars must be at least 256")
    try:
        result = _run(command, cwd=repo, timeout=timeout)
    except ToolError as exc:
        if "timed out" in str(exc):
            return TestRun("timeout", str(exc), None)
        raise
    output = (result.stdout + "\n" + result.stderr).strip()
    output = _truncate_output(output, max_output_chars)
    return TestRun("pass" if result.returncode == 0 else "fail", output, result.returncode)


def patch_fingerprint(diff_text: str) -> str:
    normalized = diff_text.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:16]


def failure_fingerprint(result: str, output: str, exit_code: int | None = None) -> str:
    """Hash failing test names, exit code, and bounded failure details."""

    salient = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR ") or stripped.startswith("E "):
            salient.append(stripped)
    if not salient:
        salient = [line.strip() for line in output.splitlines() if line.strip()][:8]
    stderr_tail = output[-1_000:]
    payload = f"exit={exit_code}\nresult={result}\n" + "\n".join(salient) + "\n" + stderr_tail
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def git_diff(repo: Path) -> str:
    repo = ensure_git_repo(repo)
    result = _run(["git", "diff", "--no-ext-diff", "--binary"], cwd=repo)
    if result.returncode != 0:
        raise ToolError(result.stderr.strip() or "git diff failed")
    return result.stdout


def reset_worktree(repo: Path) -> None:
    repo = ensure_git_repo(repo)
    reset = _run(["git", "reset", "--hard", "HEAD"], cwd=repo)
    if reset.returncode != 0:
        raise ToolError(reset.stderr.strip() or "git reset failed")
    clean = _run(["git", "clean", "-fd"], cwd=repo)
    if clean.returncode != 0:
        raise ToolError(clean.stderr.strip() or "git clean failed")
