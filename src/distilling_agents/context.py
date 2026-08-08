from __future__ import annotations

from pathlib import Path

from .tools import list_files, read_file

_TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt", ".html", ".css", ".scss", ".sql", ".go", ".rs", ".java",
}


def build_context(repo: Path, *, max_files: int = 12, max_total_chars: int = 45_000) -> str:
    """Build a deliberately small repository context pack for the worker."""

    selected: list[str] = []
    for path in list_files(repo):
        p = Path(path)
        if p.suffix.lower() in _TEXT_SUFFIXES and not any(part.startswith(".") for part in p.parts):
            selected.append(path)
        if len(selected) >= max_files:
            break

    chunks: list[str] = []
    total = 0
    for path in selected:
        body = read_file(repo, path, max_chars=8_000)
        chunk = f"\n### FILE: {path}\n{body}\n"
        if total + len(chunk) > max_total_chars:
            break
        chunks.append(chunk)
        total += len(chunk)

    file_list = "\n".join(list_files(repo))
    return f"TRACKED FILES:\n{file_list}\n\nRELEVANT FILE CONTENTS:\n{''.join(chunks)}"
