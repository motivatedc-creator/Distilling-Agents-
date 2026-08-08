from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .tools import list_files, read_file

_TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt", ".html", ".css", ".scss", ".sql", ".go", ".rs", ".java",
}
_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "does", "from",
    "have", "into", "must", "result", "should", "that", "their", "there", "these", "this",
    "when", "where", "which", "with", "wrong", "your",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class ContextPack:
    text: str
    files: tuple[str, ...]
    char_count: int
    approximate_tokens: int


def _keywords(text: str) -> tuple[str, ...]:
    words = {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}
    return tuple(sorted(word for word in words if word not in _STOPWORDS))


def _identifier_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def _score_file(path: str, body: str, keywords: tuple[str, ...]) -> int:
    path_tokens = _identifier_tokens(path.replace("/", " ").replace(".", " "))
    body_tokens = _identifier_tokens(body)
    score = 0
    for keyword in keywords:
        if keyword in path_tokens:
            score += 8
        if keyword in body_tokens:
            score += 3
    lowered = path.lower()
    if score and ("test" in lowered or "spec" in lowered):
        score += 2
    return score


def build_context(
    repo: Path,
    *,
    issue: str = "",
    failure_feedback: str = "",
    max_files: int = 4,
    max_total_chars: int = 16_000,
) -> ContextPack:
    """Build a deterministic, issue-focused context pack with a hard character budget."""

    if max_files < 1:
        raise ValueError("max_files must be at least 1")
    if max_total_chars < 256:
        raise ValueError("max_total_chars must be at least 256")

    tracked = [
        path
        for path in list_files(repo)
        if Path(path).suffix.lower() in _TEXT_SUFFIXES
        and not any(part.startswith(".") for part in Path(path).parts)
    ]
    query = f"{issue}\n{failure_feedback[:4_000]}"
    keywords = _keywords(query)

    ranked: list[tuple[int, str, str]] = []
    for path in tracked:
        body = read_file(repo, path, max_chars=min(12_000, max_total_chars))
        ranked.append((_score_file(path, body, keywords), path, body))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    matched = [item for item in ranked if item[0] > 0]
    chosen = (matched or ranked)[:max_files]

    chunks: list[str] = []
    selected_files: list[str] = []
    total = 0
    for _, path, body in chosen:
        header = f"\n### FILE: {path}\n"
        remaining = max_total_chars - total
        if remaining <= len(header):
            break
        body_budget = remaining - len(header)
        rendered_body = body[:body_budget]
        chunk = header + rendered_body
        chunks.append(chunk)
        selected_files.append(path)
        total += len(chunk)
        if total >= max_total_chars:
            break

    text = "".join(chunks)[:max_total_chars]
    # Conservative operational estimate for code-heavy prompts; the real vLLM usage
    # count is recorded by VLLMWorker after a request completes.
    approximate_tokens = (len(text) + 2) // 3
    return ContextPack(
        text=text,
        files=tuple(selected_files),
        char_count=len(text),
        approximate_tokens=approximate_tokens,
    )
