from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False
    language: str = "pst"
    model: str = "k2-fsa/OmniVoice"
    omnivoice_command: str = "omnivoice-infer"
    output_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "distilling-agents" / "voice"
    )
    speed: float = 1.0
    num_step: int = 32
    timeout_seconds: int = 300
    ref_audio: Path | None = None
    ref_text: str | None = None


@dataclass(frozen=True)
class VoiceResult:
    status: Literal["synthesized", "unavailable"]
    output_path: Path | None = None
    diagnostic: str = ""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


class OmniVoiceAdapter:
    def __init__(self, config: VoiceConfig, *, runner: Runner = subprocess.run) -> None:
        self.config = config
        self.runner = runner

    def _new_output_path(self) -> Path:
        return self.config.output_dir / f"speech-{uuid.uuid4().hex}.wav"

    def _argv(self, text: str, output_path: Path) -> list[str]:
        args = [
            self.config.omnivoice_command,
            "--model",
            self.config.model,
            "--text",
            text,
            "--language",
            self.config.language,
            "--output",
            str(output_path),
            "--num_step",
            str(self.config.num_step),
            "--speed",
            str(self.config.speed),
        ]
        if self.config.ref_audio is not None:
            args.extend(["--ref_audio", str(self.config.ref_audio)])
        if self.config.ref_text:
            args.extend(["--ref_text", self.config.ref_text])
        return args

    def synthesize(self, text: str, output_path: Path | None = None) -> VoiceResult:
        if not text.strip():
            return VoiceResult("unavailable", diagnostic="voice text is empty")

        output = output_path or self._new_output_path()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
        except OSError as exc:
            return VoiceResult(
                "unavailable",
                diagnostic=_tail(f"Could not prepare voice output path: {exc}"),
            )

        argv = self._argv(text, output)

        try:
            completed = self.runner(
                argv,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return VoiceResult("unavailable", diagnostic="OmniVoice synthesis timed out")
        except FileNotFoundError:
            return VoiceResult(
                "unavailable",
                diagnostic=f"OmniVoice executable not found: {argv[0]}",
            )
        except OSError as exc:
            return VoiceResult(
                "unavailable",
                diagnostic=_tail(f"OmniVoice process error: {exc}"),
            )

        diagnostics = _tail((completed.stdout + "\n" + completed.stderr).strip())
        if completed.returncode != 0:
            return VoiceResult(
                "unavailable",
                diagnostic=_tail(
                    f"OmniVoice exited with exit {completed.returncode}: {diagnostics}"
                ),
            )
        try:
            valid_output = output.is_file() and output.stat().st_size > 0
        except OSError as exc:
            return VoiceResult(
                "unavailable",
                diagnostic=_tail(f"Could not validate voice output: {exc}"),
            )
        if not valid_output:
            return VoiceResult(
                "unavailable",
                diagnostic="OmniVoice returned success but no non-empty WAV was produced",
            )
        return VoiceResult(
            "synthesized",
            output_path=output,
            diagnostic=diagnostics,
        )
