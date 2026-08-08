from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .playback import PlaybackAdapter
from .voice import OmniVoiceAdapter, VoiceConfig


def _default_output_dir() -> Path:
    return Path.home() / ".cache" / "distilling-agents" / "voice"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthesize speech through an isolated OmniVoice install."
    )
    p.add_argument("text")
    p.add_argument(
        "--language",
        default=os.getenv("DISTILL_VOICE_LANGUAGE", "pst"),
    )
    p.add_argument(
        "--model",
        default=os.getenv("DISTILL_OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
    )
    p.add_argument(
        "--omnivoice-command",
        default=os.getenv("DISTILL_OMNIVOICE_COMMAND", "omnivoice-infer"),
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--output-dir", type=Path, default=_default_output_dir())
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--num-step", type=int, default=32)
    p.add_argument("--ref-audio", type=Path)
    p.add_argument("--ref-text")
    p.add_argument("--play", action="store_true")
    p.add_argument("--player")
    return p


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory=OmniVoiceAdapter,
    playback_factory=PlaybackAdapter,
) -> int:
    args = parser().parse_args(argv)
    config = VoiceConfig(
        enabled=True,
        language=args.language,
        model=args.model,
        omnivoice_command=args.omnivoice_command,
        output_dir=args.output_dir,
        speed=args.speed,
        num_step=args.num_step,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
    synthesis = adapter_factory(config).synthesize(
        args.text,
        output_path=args.output,
    )
    if synthesis.status != "synthesized" or synthesis.output_path is None:
        print(
            f"Voice unavailable: {synthesis.diagnostic}",
            file=sys.stderr,
        )
        return 2

    print(f"WAV: {synthesis.output_path}")
    if args.play:
        playback = playback_factory(player=args.player).play(
            synthesis.output_path
        )
        if playback.status != "played":
            print(
                f"Playback unavailable: {playback.diagnostic}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
