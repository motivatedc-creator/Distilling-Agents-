from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .graph import run_agent
from .playback import PlaybackAdapter
from .status_voice import deliver_agent_status
from .voice import OmniVoiceAdapter, VoiceConfig
from .worker import VLLMWorker
from .worktree import temporary_worktree


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the constrained local code-repair loop.")
    p.add_argument("repo", type=Path, help="Path to a git repository to repair.")
    p.add_argument("--issue", required=True, help="Bug/issue description.")
    p.add_argument("--test-command", default="python -m pytest -q", help="Deterministic test command.")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"))
    p.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("DISTILL_VOICE", False),
    )
    p.add_argument(
        "--voice-language",
        default=os.getenv("DISTILL_VOICE_LANGUAGE", "pst"),
    )
    p.add_argument(
        "--omnivoice-command",
        default=os.getenv("DISTILL_OMNIVOICE_COMMAND", "omnivoice-infer"),
    )
    p.add_argument(
        "--voice-output-dir",
        type=Path,
        default=Path.home() / ".cache" / "distilling-agents" / "voice",
    )
    p.add_argument("--voice-player")
    p.add_argument("--voice-ref-audio", type=Path)
    p.add_argument("--voice-ref-text")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
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

    if args.voice:
        voice_config = VoiceConfig(
            enabled=True,
            language=args.voice_language,
            omnivoice_command=args.omnivoice_command,
            output_dir=args.voice_output_dir,
            ref_audio=args.voice_ref_audio,
            ref_text=args.voice_ref_text,
        )
        delivery = deliver_agent_status(
            result,
            OmniVoiceAdapter(voice_config),
            PlaybackAdapter(player=args.voice_player),
        )
        if delivery.status != "spoken":
            print(
                f"Voice: {delivery.status} — {delivery.diagnostic}",
                file=sys.stderr,
            )

    return 0 if result.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
