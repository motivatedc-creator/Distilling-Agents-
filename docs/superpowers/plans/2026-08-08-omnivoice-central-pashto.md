# OmniVoice Central Pashto Voice Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated OmniVoice speech layer to Distilling Agents with Central Pashto (`pst`) as the default, including a standalone `distill-speak` command and optional spoken coding-result notifications.

**Architecture:** OmniVoice stays in its own virtual environment and is invoked only through its `omnivoice-infer` executable. Synthesis, playback, and coding-result notification are separate modules so TTS failures cannot alter `run_agent()` behavior or benchmark results. The existing LangGraph repair loop is unchanged.

**Tech Stack:** Python 3.11+, stdlib `subprocess`/`pathlib`/`shutil`, existing project dependencies, OmniVoice 0.2.x in a separate venv, pytest, GitHub Actions.

## Global Constraints

- Default language is Central Pashto: `pst`.
- Do not add `omnivoice`, `torch`, `torchaudio`, `transformers`, or `accelerate` to the main `distilling-agents` dependencies.
- Never use `shell=True`.
- Voice text is data, never executable command text.
- Synthesis/playback failures are non-critical and never change a coding result.
- No vLLM stop/restart logic, GPU handoff supervisor, TTS server, or persistent daemon in V1.
- Voice remains downstream from `run_agent()`; `src/distilling_agents/graph.py` must not import any voice module.
- The future 10-bug benchmark must call the repair harness directly with no voice layer.
- Automatic status speech uses two deterministic Pashto templates only; no translation model.
- Voice cloning is opt-in.
- CI must not require OmniVoice, GPU access, model downloads, or a sound device.
- Default WAV directory: `~/.cache/distilling-agents/voice/`.
- Synthesis timeout: 300 seconds.
- Playback timeout: 120 seconds.
- Bound process diagnostics to the last 4,000 characters.

## File Map

- Create `src/distilling_agents/voice.py` — config/result types and OmniVoice subprocess adapter.
- Create `src/distilling_agents/playback.py` — local WAV playback adapter.
- Create `src/distilling_agents/speak.py` — standalone `distill-speak` CLI.
- Create `src/distilling_agents/status_voice.py` — deterministic Pashto status phrases and status delivery helper.
- Modify `src/distilling_agents/cli.py` — optional post-result voice notification only.
- Modify `pyproject.toml` — register `distill-speak`; no new heavy dependencies.
- Create `tests/test_voice.py`.
- Create `tests/test_playback.py`.
- Create `tests/test_speak.py`.
- Create `tests/test_status_voice.py`.
- Create `tests/test_cli_voice.py`.
- Create `docs/voice.md`.
- Modify `README.md`.

Do not modify `src/distilling_agents/graph.py`.

---

### Task 1: OmniVoice subprocess adapter

**Files:**
- Create: `src/distilling_agents/voice.py`
- Create: `tests/test_voice.py`

**Interfaces:**
- Produces `VoiceConfig`.
- Produces `VoiceResult`.
- Produces `OmniVoiceAdapter.synthesize(text: str, output_path: Path | None = None) -> VoiceResult`.

- [ ] **Step 1: Write the failing synthesis tests**

Create `tests/test_voice.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from distilling_agents.voice import OmniVoiceAdapter, VoiceConfig


class FakeRunner:
    def __init__(self, returncode: int = 0, write_output: bool = True) -> None:
        self.returncode = returncode
        self.write_output = write_output
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        argv = list(args)
        self.calls.append(argv)
        if self.write_output and "--output" in argv:
            output = Path(argv[argv.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"RIFFfake-wave")
        return subprocess.CompletedProcess(
            argv,
            self.returncode,
            "ok",
            "boom" if self.returncode else "",
        )


def test_defaults_to_central_pashto(tmp_path: Path) -> None:
    runner = FakeRunner()
    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=runner
    ).synthesize("سلام، څنګه یې؟")

    assert result.status == "synthesized"
    argv = runner.calls[0]
    assert argv[0] == "omnivoice-infer"
    assert argv[argv.index("--language") + 1] == "pst"
    assert argv[argv.index("--model") + 1] == "k2-fsa/OmniVoice"
    assert argv[argv.index("--num_step") + 1] == "32"
    assert argv[argv.index("--speed") + 1] == "1.0"
    assert result.output_path is not None
    assert result.output_path.stat().st_size > 0


def test_clone_flags_are_added_only_when_configured(tmp_path: Path) -> None:
    runner = FakeRunner()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFref")
    config = VoiceConfig(
        output_dir=tmp_path,
        ref_audio=ref,
        ref_text="زما غږ دی",
    )
    OmniVoiceAdapter(config, runner=runner).synthesize("سلام")

    argv = runner.calls[0]
    assert argv[argv.index("--ref_audio") + 1] == str(ref)
    assert argv[argv.index("--ref_text") + 1] == "زما غږ دی"


def test_nonzero_exit_is_unavailable(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=7, write_output=False)
    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=runner
    ).synthesize("سلام")
    assert result.status == "unavailable"
    assert "exit 7" in result.diagnostic.lower()


def test_missing_wav_is_unavailable(tmp_path: Path) -> None:
    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=FakeRunner(write_output=False)
    ).synthesize("سلام")
    assert result.status == "unavailable"
    assert "wav" in result.diagnostic.lower()


def test_timeout_is_unavailable(tmp_path: Path) -> None:
    def timeout_runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout=1)

    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=timeout_runner
    ).synthesize("سلام")
    assert result.status == "unavailable"
    assert "timed out" in result.diagnostic.lower()


def test_missing_executable_is_unavailable(tmp_path: Path) -> None:
    def missing_runner(args, **kwargs):
        raise FileNotFoundError(args[0])

    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=missing_runner
    ).synthesize("سلام")
    assert result.status == "unavailable"
    assert "not found" in result.diagnostic.lower()


def test_diagnostic_is_bounded(tmp_path: Path) -> None:
    def loud_runner(args, **kwargs):
        return subprocess.CompletedProcess(list(args), 1, "x" * 8000, "TAIL")

    result = OmniVoiceAdapter(
        VoiceConfig(output_dir=tmp_path), runner=loud_runner
    ).synthesize("سلام")
    assert len(result.diagnostic) <= 4000
    assert "TAIL" in result.diagnostic
```

- [ ] **Step 2: Run the tests and verify red state**

Run:

```bash
pytest tests/test_voice.py -v
```

Expected: import/collection failure because `distilling_agents.voice` does not exist.

- [ ] **Step 3: Implement `voice.py` exactly at the external-process boundary**

Create `src/distilling_agents/voice.py`:

```python
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
        default_factory=lambda: Path.home()
        / ".cache"
        / "distilling-agents"
        / "voice"
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
    def __init__(
        self,
        config: VoiceConfig,
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self.config = config
        self.runner = runner

    def _new_output_path(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
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

    def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> VoiceResult:
        if not text.strip():
            return VoiceResult("unavailable", diagnostic="voice text is empty")

        output = output_path or self._new_output_path()
        output.parent.mkdir(parents=True, exist_ok=True)
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
            return VoiceResult(
                "unavailable",
                diagnostic="OmniVoice synthesis timed out",
            )
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

        process_output = _tail(
            (completed.stdout + "\n" + completed.stderr).strip()
        )
        if completed.returncode != 0:
            return VoiceResult(
                "unavailable",
                diagnostic=_tail(
                    f"OmniVoice exited with exit {completed.returncode}: "
                    f"{process_output}"
                ),
            )
        if not output.is_file() or output.stat().st_size == 0:
            return VoiceResult(
                "unavailable",
                diagnostic=(
                    "OmniVoice returned success but no non-empty WAV "
                    "was produced"
                ),
            )
        return VoiceResult(
            "synthesized",
            output_path=output,
            diagnostic=process_output,
        )
```

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_voice.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/distilling_agents/voice.py tests/test_voice.py
git commit -m "feat: add isolated OmniVoice synthesis adapter"
```

---

### Task 2: Best-effort local playback

**Files:**
- Create: `src/distilling_agents/playback.py`
- Create: `tests/test_playback.py`

**Interfaces:**
- Produces `PlaybackResult`.
- Produces `PlaybackAdapter.play(wav_path: Path) -> PlaybackResult`.
- Does not import OmniVoice or `voice.py`.

- [ ] **Step 1: Write failing playback tests**

Create `tests/test_playback.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from distilling_agents.playback import PlaybackAdapter


def test_ffplay_uses_noninteractive_flags(tmp_path: Path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    calls = []

    def which(name: str):
        return "/usr/bin/ffplay" if name == "ffplay" else None

    def runner(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, "", "")

    result = PlaybackAdapter(which=which, runner=runner).play(wav)
    assert result.status == "played"
    assert calls[0] == [
        "/usr/bin/ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        str(wav),
    ]


def test_missing_player_is_unavailable(tmp_path: Path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    result = PlaybackAdapter(which=lambda _: None).play(wav)
    assert result.status == "unavailable"
    assert "player" in result.diagnostic.lower()


def test_player_failure_is_unavailable(tmp_path: Path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")

    def runner(args, **kwargs):
        return subprocess.CompletedProcess(
            list(args), 2, "", "audio device busy"
        )

    result = PlaybackAdapter(
        player="/usr/bin/aplay", runner=runner
    ).play(wav)
    assert result.status == "unavailable"
    assert "audio device busy" in result.diagnostic
```

- [ ] **Step 2: Run red test**

```bash
pytest tests/test_playback.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement playback adapter with no shell**

Create `src/distilling_agents/playback.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


@dataclass(frozen=True)
class PlaybackResult:
    status: Literal["played", "unavailable"]
    diagnostic: str = ""


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


class PlaybackAdapter:
    def __init__(
        self,
        *,
        player: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner=subprocess.run,
        timeout_seconds: int = 120,
    ) -> None:
        self.player = player
        self.which = which
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def _resolve_player(self) -> str | None:
        if self.player:
            return self.player
        for name in ("ffplay", "paplay", "aplay"):
            found = self.which(name)
            if found:
                return found
        return None

    def _argv(self, player: str, wav_path: Path) -> list[str]:
        if Path(player).name.lower().startswith("ffplay"):
            return [
                player,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                str(wav_path),
            ]
        return [player, str(wav_path)]

    def play(self, wav_path: Path) -> PlaybackResult:
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            return PlaybackResult(
                "unavailable", "WAV file is missing or empty"
            )
        player = self._resolve_player()
        if player is None:
            return PlaybackResult(
                "unavailable", "No supported local audio player found"
            )
        argv = self._argv(player, wav_path)
        try:
            completed = self.runner(
                argv,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PlaybackResult("unavailable", "Audio playback timed out")
        except FileNotFoundError:
            return PlaybackResult(
                "unavailable", f"Audio player not found: {player}"
            )
        except OSError as exc:
            return PlaybackResult(
                "unavailable", _tail(f"Playback process error: {exc}")
            )

        diagnostic = _tail(
            (completed.stdout + "\n" + completed.stderr).strip()
        )
        if completed.returncode != 0:
            return PlaybackResult("unavailable", diagnostic)
        return PlaybackResult("played", diagnostic)
```

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_playback.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/distilling_agents/playback.py tests/test_playback.py
git commit -m "feat: add best-effort local voice playback"
```

---

### Task 3: Standalone `distill-speak` CLI

**Files:**
- Create: `src/distilling_agents/speak.py`
- Create: `tests/test_speak.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes `VoiceConfig`, `OmniVoiceAdapter`, `PlaybackAdapter`.
- Produces console script `distill-speak = "distilling_agents.speak:main"`.
- Exit `0`: synthesis succeeded.
- Exit `2`: synthesis unavailable.
- Playback failure never changes a successful synthesis exit code.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_speak.py`:

```python
from pathlib import Path

from distilling_agents.playback import PlaybackResult
from distilling_agents.speak import main, parser
from distilling_agents.voice import VoiceResult


class FakeVoice:
    def __init__(self, config):
        self.config = config

    def synthesize(self, text, output_path=None):
        output = output_path or self.config.output_dir / "fake.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")
        return VoiceResult("synthesized", output_path=output)


class FailedVoice:
    def __init__(self, config):
        pass

    def synthesize(self, text, output_path=None):
        return VoiceResult(
            "unavailable", diagnostic="cuda out of memory"
        )


class FailedPlayback:
    def __init__(self, **kwargs):
        pass

    def play(self, wav_path):
        return PlaybackResult("unavailable", "no sound device")


def test_parser_defaults_to_central_pashto() -> None:
    args = parser().parse_args(["سلام"])
    assert args.language == "pst"
    assert args.model == "k2-fsa/OmniVoice"
    assert args.play is False


def test_success_returns_zero(tmp_path: Path, capsys) -> None:
    code = main(
        ["سلام", "--output-dir", str(tmp_path)],
        adapter_factory=FakeVoice,
    )
    assert code == 0
    assert "fake.wav" in capsys.readouterr().out


def test_playback_failure_does_not_change_success_exit(
    tmp_path: Path,
) -> None:
    code = main(
        ["سلام", "--output-dir", str(tmp_path), "--play"],
        adapter_factory=FakeVoice,
        playback_factory=FailedPlayback,
    )
    assert code == 0


def test_synthesis_failure_returns_two(tmp_path: Path, capsys) -> None:
    code = main(
        ["سلام", "--output-dir", str(tmp_path)],
        adapter_factory=FailedVoice,
    )
    assert code == 2
    assert "cuda out of memory" in capsys.readouterr().err
```

- [ ] **Step 2: Run red test**

```bash
pytest tests/test_speak.py -v
```

Expected: import failure because `speak.py` does not exist.

- [ ] **Step 3: Implement parser and execution path**

Create `src/distilling_agents/speak.py`:

```python
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
```

- [ ] **Step 4: Register script only; add no OmniVoice dependency**

Modify `pyproject.toml`:

```toml
[project.scripts]
distill-agent = "distilling_agents.cli:main"
distill-speak = "distilling_agents.speak:main"
```

- [ ] **Step 5: Run focused tests and CLI metadata check**

```bash
pytest tests/test_speak.py -v
python -m pip install -e '.[dev]'
distill-speak --help
```

Expected: tests pass; help works even when OmniVoice itself is not installed.

- [ ] **Step 6: Commit**

```bash
git add src/distilling_agents/speak.py tests/test_speak.py pyproject.toml
git commit -m "feat: add Central Pashto distill-speak command"
```

---

### Task 4: Optional spoken coding-result status

**Files:**
- Create: `src/distilling_agents/status_voice.py`
- Create: `tests/test_status_voice.py`
- Create: `tests/test_cli_voice.py`
- Modify: `src/distilling_agents/cli.py`

**Interfaces:**
- Produces `status_text(status: str) -> str`.
- Produces `StatusVoiceResult`.
- Produces `deliver_agent_status(result, voice_adapter, playback_adapter) -> StatusVoiceResult`.
- Coding exit status remains based only on `AgentResult.status`.

- [ ] **Step 1: Write failing status-delivery tests**

Create `tests/test_status_voice.py`:

```python
from distilling_agents.models import AgentResult
from distilling_agents.playback import PlaybackResult
from distilling_agents.status_voice import deliver_agent_status, status_text
from distilling_agents.voice import VoiceResult


class FailedVoice:
    def synthesize(self, text, output_path=None):
        return VoiceResult("unavailable", diagnostic="oom")


class FakeVoice:
    def __init__(self, output_path):
        self.output_path = output_path

    def synthesize(self, text, output_path=None):
        return VoiceResult("synthesized", output_path=self.output_path)


class FakePlayback:
    def __init__(self, status="played"):
        self.status = status

    def play(self, wav_path):
        return PlaybackResult(
            self.status,
            "no device" if self.status == "unavailable" else "",
        )


def test_status_text_is_deterministic_pashto() -> None:
    assert status_text("passed") == "کار بشپړ شو، ټول ټېسټونه پاس شول."
    assert status_text("blocked") == "کار ودرېد، حل تر اوسه پیدا نه شو."


def test_voice_failure_does_not_mutate_passed_result(tmp_path) -> None:
    agent = AgentResult(status="passed", attempts=1)
    delivery = deliver_agent_status(
        agent,
        FailedVoice(),
        FakePlayback(),
    )
    assert agent.status == "passed"
    assert delivery.status == "unavailable"


def test_playback_failure_is_synthesized_only(tmp_path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    delivery = deliver_agent_status(
        AgentResult(status="blocked", attempts=3),
        FakeVoice(wav),
        FakePlayback("unavailable"),
    )
    assert delivery.status == "synthesized-only"
```

- [ ] **Step 2: Run red test**

```bash
pytest tests/test_status_voice.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement deterministic status delivery**

Create `src/distilling_agents/status_voice.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import AgentResult


_STATUS_TEXT = {
    "passed": "کار بشپړ شو، ټول ټېسټونه پاس شول.",
    "blocked": "کار ودرېد، حل تر اوسه پیدا نه شو.",
}


@dataclass(frozen=True)
class StatusVoiceResult:
    status: Literal["spoken", "synthesized-only", "unavailable"]
    diagnostic: str = ""


def status_text(status: str) -> str:
    return _STATUS_TEXT.get(status, _STATUS_TEXT["blocked"])


def deliver_agent_status(
    result: AgentResult,
    voice_adapter,
    playback_adapter,
) -> StatusVoiceResult:
    synthesis = voice_adapter.synthesize(status_text(result.status))
    if synthesis.status != "synthesized" or synthesis.output_path is None:
        return StatusVoiceResult(
            "unavailable",
            synthesis.diagnostic,
        )

    playback = playback_adapter.play(synthesis.output_path)
    if playback.status != "played":
        return StatusVoiceResult(
            "synthesized-only",
            playback.diagnostic,
        )
    return StatusVoiceResult("spoken", playback.diagnostic)
```

- [ ] **Step 4: Run status tests**

```bash
pytest tests/test_status_voice.py -v
```

Expected: all pass.

- [ ] **Step 5: Add voice CLI flags while keeping default off**

Modify `src/distilling_agents/cli.py`.

Add imports:

```python
import sys

from .playback import PlaybackAdapter
from .status_voice import deliver_agent_status
from .voice import OmniVoiceAdapter, VoiceConfig
```

Add helper:

```python
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
```

Add parser arguments:

```python
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
```

Change `main` signature only:

```python
def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
```

Keep all existing repair logic unchanged.

- [ ] **Step 6: Add post-result notification with exit-code isolation**

After leaving `temporary_worktree` and after printing the existing JSON result, add:

```python
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
```

Do not pass `result.diff` or `result.error_log` into TTS.

- [ ] **Step 7: Write CLI regression tests**

Create `tests/test_cli_voice.py`:

```python
from distilling_agents.cli import parser


def test_existing_cli_defaults_voice_off() -> None:
    args = parser().parse_args(["/tmp/repo", "--issue", "fix it"])
    assert args.voice is False
    assert args.voice_language == "pst"


def test_voice_can_be_enabled_explicitly() -> None:
    args = parser().parse_args([
        "/tmp/repo",
        "--issue",
        "fix it",
        "--voice",
    ])
    assert args.voice is True
```

Then add one integration-style unit test by monkeypatching `temporary_worktree`, `run_agent`, `OmniVoiceAdapter`, and `PlaybackAdapter` so a fake voice failure occurs after a fake `AgentResult(status="passed", attempts=1)`. Assert `main(...) == 0`. Repeat with `status="blocked"` and assert `main(...) == 2`.

Use concrete fakes:

```python
class FailedAdapter:
    def __init__(self, config):
        pass

    def synthesize(self, text, output_path=None):
        from distilling_agents.voice import VoiceResult
        return VoiceResult("unavailable", diagnostic="oom")


class DummyPlayback:
    def __init__(self, **kwargs):
        pass
```

The test must not start vLLM or OmniVoice.

- [ ] **Step 8: Run focused and full tests**

```bash
pytest tests/test_status_voice.py tests/test_cli_voice.py -v
pytest
```

Expected: all pass; all pre-existing repair-loop tests remain green.

- [ ] **Step 9: Commit**

```bash
git add src/distilling_agents/status_voice.py src/distilling_agents/cli.py tests/test_status_voice.py tests/test_cli_voice.py
git commit -m "feat: speak optional Pashto agent status"
```

---

### Task 5: Documentation and real Central Pashto acceptance

**Files:**
- Create: `docs/voice.md`
- Modify: `README.md`

**Interfaces:**
- Documentation only.
- Real GPU/audio testing happens here, never in CI.

- [ ] **Step 1: Write isolated WSL2 setup documentation**

Create `docs/voice.md` with this environment boundary:

```text
Distilling-Agents-/.venv/          # coding harness
~/.local/omnivoice-venv/           # OmniVoice + Torch/audio stack
```

Document the integration baseline commands from upstream OmniVoice:

```bash
python3 -m venv ~/.local/omnivoice-venv
source ~/.local/omnivoice-venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install omnivoice==0.2.1
```

State that these versions are the integration baseline and must be rechecked against upstream before future upgrades.

- [ ] **Step 2: Document standalone synthesis, playback, and cloning**

Include exactly these examples:

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --play
```

```bash
distill-speak "کار بشپړ شو" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --ref-audio /path/to/pashto-reference.wav \
  --ref-text "د ريفرنس غږ متن"
```

State: only clone audio the user has permission to use.

- [ ] **Step 3: Document coding-agent voice and GPU contention**

Include:

```bash
distill-agent /path/to/repo \
  --issue "fix the bug" \
  --voice \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

Explain that on an 8 GB single GPU, a running Qwen/vLLM server may occupy too much VRAM for OmniVoice. V1 reports voice as unavailable and never kills/restarts vLLM. For manual voice testing, stop vLLM first if CUDA OOM occurs.

- [ ] **Step 4: Document playback setup for WSL2**

State detection order: `ffplay`, `paplay`, `aplay`.

Provide one concrete setup path:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Manual player check:

```bash
ffplay -nodisp -autoexit /tmp/distilling-agents-pst.wav
```

- [ ] **Step 5: Add concise README entry**

Add:

```markdown
## Central Pashto voice

Distilling Agents can optionally synthesize status speech through an isolated OmniVoice installation. Central Pashto (`pst`) is the default language.

```bash
distill-speak "سلام، څنګه یې؟" --play
```

The coding harness does not install or import OmniVoice directly. See `docs/voice.md` for WSL2/CUDA setup, voice cloning, playback, and GPU-contention notes. Voice is disabled for the coding benchmark.
```

- [ ] **Step 6: Run software verification without OmniVoice installed**

```bash
python -m pip install -e '.[dev]'
distill-speak --help
distill-agent --help
pytest
```

Expected: both CLIs load; full tests pass; no OmniVoice model is downloaded.

- [ ] **Step 7: Run real target-machine Central Pashto acceptance**

With the dedicated OmniVoice environment installed and vLLM stopped if VRAM requires it:

```bash
distill-speak "سلام، څنګه یې؟ کار بشپړ شو." \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --output /tmp/distilling-agents-pst.wav \
  --play
```

Acceptance requirements:

1. command exits `0`;
2. `/tmp/distilling-agents-pst.wav` exists and is non-empty;
3. audible speech is produced through a supported player;
4. the user recognizes the speech as Central Pashto;
5. no coding-agent/vLLM process is modified or terminated by the voice layer.

If the WAV is valid but pronunciation/voice quality is poor, record that as a model-quality finding, not a harness failure.

- [ ] **Step 8: Run CI and commit docs**

```bash
git add docs/voice.md README.md
git commit -m "docs: document Central Pashto voice setup"
git push
```

Require the Python 3.11 GitHub Actions `pytest` job to pass. CI must not install OmniVoice or download model weights.

---

## Final Verification

Run:

```bash
pytest
python -m pip install -e '.[dev]'
distill-speak --help
distill-agent --help
```

Review assertions:

- `src/distilling_agents/graph.py` has no voice imports or TTS calls.
- `pyproject.toml` has no OmniVoice/Torch dependency additions.
- all subprocesses use argv lists and never `shell=True`.
- process errors become structured voice/playback results.
- coding CLI exit status depends only on `AgentResult.status`.
- automatic speech never receives diffs, logs, SHAs, filenames, or raw test output.
- `pst` is the default language everywhere unless explicitly overridden.
- coding CLI voice defaults off.
- `distill-speak` can generate a WAV without playback.
- the benchmark remains independent from voice.

Do not claim real Central Pashto audio works end-to-end until Task 5 Step 7 succeeds on the target machine.
