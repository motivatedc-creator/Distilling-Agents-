# OmniVoice Central Pashto Voice Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated OmniVoice speech layer to Distilling Agents with Central Pashto (`pst`) as the default, including a standalone `distill-speak` command and optional spoken coding-result notifications.

**Architecture:** Keep OmniVoice out of the main Python environment and call its `omnivoice-infer` executable through a narrow subprocess adapter. Keep synthesis, playback, and coding-agent status notification as separate units so TTS failures cannot affect `run_agent()` or benchmark results. The existing LangGraph repair loop remains unchanged.

**Tech Stack:** Python 3.11+, stdlib `subprocess`/`pathlib`/`shutil`, existing Pydantic dependency only where already useful, OmniVoice 0.2.x in a separate virtual environment, pytest, GitHub Actions.

## Global Constraints

- Central Pashto is the default voice language: `pst`.
- OmniVoice remains an external executable integration; do not add `omnivoice`, `torch`, `torchaudio`, `transformers`, or `accelerate` to the main `distilling-agents` dependency list.
- Never use `shell=True`.
- Voice text is data only and must never become executable command text.
- Voice synthesis and playback are non-critical; coding results stay `passed` or `blocked` regardless of voice failures.
- Do not add GPU model handoff, vLLM shutdown/restart, a persistent TTS daemon, or a TTS server in V1.
- Automatic coding-result voice is downstream from `run_agent()` and must not be imported into `graph.py`.
- The future 10-bug benchmark must remain voice-free by calling the repair harness directly, not the spoken-notification CLI path.
- Automatic status speech uses deterministic curated Pashto templates; do not add a translation model.
- Voice cloning is opt-in through explicit reference audio/text arguments.
- Core CI must not require OmniVoice, a GPU, model downloads, or a real sound device.
- Default generated WAV directory: `~/.cache/distilling-agents/voice/`.
- Default synthesis timeout: 300 seconds.
- Bound captured OmniVoice/playback diagnostics to the final 4,000 characters.

---

## File Structure

Create or modify these files only for V1:

- `src/distilling_agents/voice.py` — `VoiceConfig`, structured synthesis result, OmniVoice subprocess adapter, output-path creation.
- `src/distilling_agents/playback.py` — local WAV playback detection/execution only.
- `src/distilling_agents/speak.py` — standalone `distill-speak` CLI.
- `src/distilling_agents/status_voice.py` — deterministic Pashto status phrases and delivery helper for agent results.
- `src/distilling_agents/cli.py` — add optional voice flags and call status delivery after the coding result is finalized.
- `pyproject.toml` — register `distill-speak`; do not add OmniVoice dependencies.
- `tests/test_voice.py` — synthesis adapter contract and failure isolation.
- `tests/test_playback.py` — local playback contract.
- `tests/test_speak.py` — standalone CLI behavior.
- `tests/test_status_voice.py` — Pashto status mapping and non-critical delivery behavior.
- `tests/test_cli_voice.py` — existing coding CLI remains backward compatible and voice cannot change its exit code.
- `docs/voice.md` — isolated OmniVoice setup and manual acceptance procedure.
- `README.md` — short usage section linking to `docs/voice.md`.

Do not modify `src/distilling_agents/graph.py` for this feature.

---

### Task 1: Add the isolated OmniVoice synthesis adapter

**Files:**
- Create: `src/distilling_agents/voice.py`
- Create: `tests/test_voice.py`

**Interfaces:**
- Produces: `VoiceConfig`, `VoiceResult`, `OmniVoiceAdapter.synthesize(text: str, output_path: Path | None = None) -> VoiceResult`.
- Later tasks consume the exact classes above.

- [ ] **Step 1: Write failing tests for Central Pashto defaults and exact argument construction**

Create `tests/test_voice.py` with a fake runner that records argv and writes fake WAV bytes to the path following `--output`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from distilling_agents.voice import OmniVoiceAdapter, VoiceConfig


class FakeRunner:
    def __init__(self, *, returncode: int = 0, write_output: bool = True) -> None:
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
        return subprocess.CompletedProcess(argv, self.returncode, "ok", "boom" if self.returncode else "")


def test_synthesis_defaults_to_central_pashto(tmp_path: Path) -> None:
    runner = FakeRunner()
    config = VoiceConfig(output_dir=tmp_path)
    result = OmniVoiceAdapter(config, runner=runner).synthesize("سلام، څنګه یې؟")

    assert result.status == "synthesized"
    argv = runner.calls[0]
    assert argv[0] == "omnivoice-infer"
    assert argv[argv.index("--language") + 1] == "pst"
    assert argv[argv.index("--model") + 1] == "k2-fsa/OmniVoice"
    assert argv[argv.index("--num_step") + 1] == "32"
    assert argv[argv.index("--speed") + 1] == "1.0"
    assert result.output_path is not None
    assert result.output_path.exists()


def test_voice_clone_flags_are_only_added_when_configured(tmp_path: Path) -> None:
    runner = FakeRunner()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFref")
    config = VoiceConfig(output_dir=tmp_path, ref_audio=ref, ref_text="زما غږ دی")
    OmniVoiceAdapter(config, runner=runner).synthesize("سلام")

    argv = runner.calls[0]
    assert argv[argv.index("--ref_audio") + 1] == str(ref)
    assert argv[argv.index("--ref_text") + 1] == "زما غږ دی"
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the voice module does not exist**

Run:

```bash
pytest tests/test_voice.py -v
```

Expected: collection/import failure for `distilling_agents.voice`.

- [ ] **Step 3: Implement the minimal domain types and argv builder**

Create `src/distilling_agents/voice.py` with these exact public types:

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


class OmniVoiceAdapter:
    def __init__(self, config: VoiceConfig, *, runner: Runner = subprocess.run) -> None:
        self.config = config
        self.runner = runner

    def _new_output_path(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return self.config.output_dir / f"speech-{uuid.uuid4().hex}.wav"

    def _argv(self, text: str, output_path: Path) -> list[str]:
        args = [
            self.config.omnivoice_command,
            "--model", self.config.model,
            "--text", text,
            "--language", self.config.language,
            "--output", str(output_path),
            "--num_step", str(self.config.num_step),
            "--speed", str(self.config.speed),
        ]
        if self.config.ref_audio is not None:
            args.extend(["--ref_audio", str(self.config.ref_audio)])
        if self.config.ref_text:
            args.extend(["--ref_text", self.config.ref_text])
        return args
```

Then implement `synthesize()` in the next step rather than adding speculative behavior now.

- [ ] **Step 4: Write failing tests for all expected process failures**

Append tests covering:

```python
import pytest


def test_nonzero_exit_returns_unavailable(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=7, write_output=False)
    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=runner).synthesize("سلام")
    assert result.status == "unavailable"
    assert "exit 7" in result.diagnostic.lower()


def test_missing_wav_returns_unavailable(tmp_path: Path) -> None:
    runner = FakeRunner(write_output=False)
    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=runner).synthesize("سلام")
    assert result.status == "unavailable"
    assert "wav" in result.diagnostic.lower()


def test_timeout_returns_unavailable(tmp_path: Path) -> None:
    def timeout_runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout=1)

    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=timeout_runner).synthesize("سلام")
    assert result.status == "unavailable"
    assert "timed out" in result.diagnostic.lower()


def test_missing_executable_returns_unavailable(tmp_path: Path) -> None:
    def missing_runner(args, **kwargs):
        raise FileNotFoundError(args[0])

    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=missing_runner).synthesize("سلام")
    assert result.status == "unavailable"
    assert "not found" in result.diagnostic.lower()
```

Also test diagnostics are bounded:

```python
def test_diagnostic_is_bounded(tmp_path: Path) -> None:
    class LoudRunner(FakeRunner):
        def __call__(self, args, **kwargs):
            return subprocess.CompletedProcess(list(args), 1, "x" * 8000, "TAIL")

    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=LoudRunner()).synthesize("سلام")
    assert len(result.diagnostic) <= 4000
    assert "TAIL" in result.diagnostic
```

- [ ] **Step 5: Run the tests and confirm the failure-path tests are red**

Run:

```bash
pytest tests/test_voice.py -v
```

Expected: default argv tests may pass after Step 3, failure-path tests fail because `synthesize()` is not complete.

- [ ] **Step 6: Implement `synthesize()` with a strict exception boundary**

Add:

```python
def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def synthesize(self, text: str, output_path: Path | None = None) -> VoiceResult:
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
        return VoiceResult("unavailable", diagnostic="OmniVoice synthesis timed out")
    except FileNotFoundError:
        return VoiceResult("unavailable", diagnostic=f"OmniVoice executable not found: {argv[0]}")
    except OSError as exc:
        return VoiceResult("unavailable", diagnostic=_tail(f"OmniVoice process error: {exc}"))

    diagnostics = _tail((completed.stdout + "\n" + completed.stderr).strip())
    if completed.returncode != 0:
        return VoiceResult(
            "unavailable",
            diagnostic=_tail(f"OmniVoice exited with exit {completed.returncode}: {diagnostics}"),
        )
    if not output.is_file() or output.stat().st_size == 0:
        return VoiceResult("unavailable", diagnostic="OmniVoice returned success but no non-empty WAV was produced")
    return VoiceResult("synthesized", output_path=output, diagnostic=diagnostics)
```

Keep all methods inside `OmniVoiceAdapter`; do not invoke OmniVoice through a shell.

- [ ] **Step 7: Run the synthesis tests**

Run:

```bash
pytest tests/test_voice.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the synthesis adapter**

```bash
git add src/distilling_agents/voice.py tests/test_voice.py
git commit -m "feat: add isolated OmniVoice synthesis adapter"
```

---

### Task 2: Add best-effort local WAV playback

**Files:**
- Create: `src/distilling_agents/playback.py`
- Create: `tests/test_playback.py`

**Interfaces:**
- Consumes: a harness-controlled `Path` to a WAV file.
- Produces: `PlaybackResult` and `PlaybackAdapter.play(wav_path: Path) -> PlaybackResult`.
- Does not depend on OmniVoice or import `voice.py`.

- [ ] **Step 1: Write failing tests for player detection, success, and unavailable states**

Create tests using injected `which` and `runner` functions:

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
        "/usr/bin/ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(wav)
    ]


def test_no_local_player_returns_unavailable(tmp_path: Path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    result = PlaybackAdapter(which=lambda _: None).play(wav)
    assert result.status == "unavailable"
    assert "player" in result.diagnostic.lower()


def test_playback_nonzero_exit_is_nonfatal_result(tmp_path: Path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")

    def runner(args, **kwargs):
        return subprocess.CompletedProcess(list(args), 2, "", "audio device busy")

    result = PlaybackAdapter(player="/usr/bin/aplay", runner=runner).play(wav)
    assert result.status == "unavailable"
    assert "audio device busy" in result.diagnostic
```

- [ ] **Step 2: Run tests and verify the missing module fails**

```bash
pytest tests/test_playback.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement the playback adapter**

Create `src/distilling_agents/playback.py` with:

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
        name = Path(player).name.lower()
        if name.startswith("ffplay"):
            return [player, "-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)]
        return [player, str(wav_path)]
```

Implement `play()` with the same failure philosophy as the synthesis adapter: validate the WAV exists, catch `TimeoutExpired`, `FileNotFoundError`, and `OSError`, bound output to 4,000 characters, and return `unavailable` rather than raising.

- [ ] **Step 4: Run playback tests**

```bash
pytest tests/test_playback.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit playback support**

```bash
git add src/distilling_agents/playback.py tests/test_playback.py
git commit -m "feat: add best-effort local voice playback"
```

---

### Task 3: Add the standalone `distill-speak` command

**Files:**
- Create: `src/distilling_agents/speak.py`
- Create: `tests/test_speak.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `VoiceConfig`, `OmniVoiceAdapter`, `PlaybackAdapter`.
- Produces: package script `distill-speak = "distilling_agents.speak:main"`.
- Exit code `0` means synthesis succeeded; exit code `2` means synthesis was unavailable. Playback failure never changes a successful synthesis exit code.

- [ ] **Step 1: Write failing parser tests for Central Pashto defaults and clone flags**

Create `tests/test_speak.py`:

```python
from distilling_agents.speak import parser


def test_speak_parser_defaults_to_central_pashto() -> None:
    args = parser().parse_args(["سلام"])
    assert args.text == "سلام"
    assert args.language == "pst"
    assert args.model == "k2-fsa/OmniVoice"
    assert args.play is False


def test_speak_parser_accepts_voice_clone_inputs() -> None:
    args = parser().parse_args([
        "سلام",
        "--ref-audio", "/tmp/ref.wav",
        "--ref-text", "زما غږ دی",
    ])
    assert str(args.ref_audio) == "/tmp/ref.wav"
    assert args.ref_text == "زما غږ دی"
```

- [ ] **Step 2: Run tests and confirm the CLI module is missing**

```bash
pytest tests/test_speak.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement the parser and config builder**

`src/distilling_agents/speak.py` must expose `parser()` and `main(argv: list[str] | None = None) -> int`.

Parser arguments:

```text
text                     positional
--language               default env DISTILL_VOICE_LANGUAGE or pst
--model                  default env DISTILL_OMNIVOICE_MODEL or k2-fsa/OmniVoice
--omnivoice-command      default env DISTILL_OMNIVOICE_COMMAND or omnivoice-infer
--output                 Path or None
--output-dir             default ~/.cache/distilling-agents/voice
--speed                  float, default 1.0
--num-step               int, default 32
--ref-audio              Path or None
--ref-text               str or None
--play                   store_true
--player                 optional executable path
```

Construct `VoiceConfig(enabled=True, ...)` from those values. Do not import OmniVoice itself.

- [ ] **Step 4: Write failing execution tests using fakes**

Design `main()` to accept optional factories as keyword-only test seams without changing console usage:

```python
def main(
    argv: list[str] | None = None,
    *,
    adapter_factory=OmniVoiceAdapter,
    playback_factory=PlaybackAdapter,
) -> int:
    ...
```

Add tests proving:

```python
from pathlib import Path
from distilling_agents.voice import VoiceResult
from distilling_agents.playback import PlaybackResult
from distilling_agents.speak import main


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
        return VoiceResult("unavailable", diagnostic="cuda out of memory")


class FailedPlayback:
    def __init__(self, **kwargs):
        pass

    def play(self, wav_path):
        return PlaybackResult("unavailable", "no sound device")


def test_speak_returns_zero_when_synthesis_succeeds(tmp_path, capsys) -> None:
    code = main(["سلام", "--output-dir", str(tmp_path)], adapter_factory=FakeVoice)
    assert code == 0
    assert "fake.wav" in capsys.readouterr().out


def test_playback_failure_does_not_change_success_exit(tmp_path) -> None:
    code = main(
        ["سلام", "--output-dir", str(tmp_path), "--play"],
        adapter_factory=FakeVoice,
        playback_factory=FailedPlayback,
    )
    assert code == 0


def test_synthesis_failure_returns_two(tmp_path, capsys) -> None:
    code = main(["سلام", "--output-dir", str(tmp_path)], adapter_factory=FailedVoice)
    assert code == 2
    assert "cuda out of memory" in capsys.readouterr().err
```

- [ ] **Step 5: Implement `main()` and keep output machine-readable enough to debug**

Behavior:

1. Parse args.
2. Build `VoiceConfig`.
3. Call `adapter.synthesize()`.
4. On unavailable, print `Voice unavailable: <diagnostic>` to stderr and return `2`.
5. On synthesized, print `WAV: <path>`.
6. If `--play`, call playback; if playback unavailable print `Playback unavailable: <diagnostic>` to stderr but still return `0`.

- [ ] **Step 6: Register the package script without adding dependencies**

Modify only the scripts section of `pyproject.toml`:

```toml
[project.scripts]
distill-agent = "distilling_agents.cli:main"
distill-speak = "distilling_agents.speak:main"
```

Do not add an OmniVoice optional dependency; isolation is intentional.

- [ ] **Step 7: Run standalone CLI tests and package metadata checks**

```bash
pytest tests/test_speak.py -v
python -m pip install -e '.[dev]'
distill-speak --help
```

Expected: tests pass; help displays Central Pashto voice options; no OmniVoice import error occurs merely by showing help.

- [ ] **Step 8: Commit the standalone speech command**

```bash
git add src/distilling_agents/speak.py tests/test_speak.py pyproject.toml
git commit -m "feat: add Central Pashto distill-speak command"
```

---

### Task 4: Add non-critical spoken status notifications to the coding CLI

**Files:**
- Create: `src/distilling_agents/status_voice.py`
- Create: `tests/test_status_voice.py`
- Create: `tests/test_cli_voice.py`
- Modify: `src/distilling_agents/cli.py`

**Interfaces:**
- Consumes: `AgentResult`, `VoiceConfig`, `OmniVoiceAdapter`, `PlaybackAdapter`.
- Produces: `status_text(status: str) -> str` and `deliver_agent_status(...) -> StatusVoiceResult`.
- `graph.py` and `run_agent()` remain voice-unaware.

- [ ] **Step 1: Write failing tests for deterministic Pashto status text**

Create `tests/test_status_voice.py`:

```python
from distilling_agents.status_voice import status_text


def test_passed_status_is_central_pashto_template() -> None:
    assert status_text("passed") == "کار بشپړ شو، ټول ټېسټونه پاس شول."


def test_blocked_status_is_central_pashto_template() -> None:
    assert status_text("blocked") == "کار ودرېد، حل تر اوسه پیدا نه شو."
```

Keep exactly two V1 phrases. Do not add dynamic translation or technical details.

- [ ] **Step 2: Run tests and verify the status module is missing**

```bash
pytest tests/test_status_voice.py -v
```

Expected: import failure.

- [ ] **Step 3: Implement status mapping and structured delivery result**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import AgentResult
from .playback import PlaybackAdapter, PlaybackResult
from .voice import OmniVoiceAdapter, VoiceResult


_STATUS_TEXT = {
    "passed": "کار بشپړ شو، ټول ټېسټونه پاس شول.",
    "blocked": "کار ودرېد، حل تر اوسه پیدا نه شو.",
}


def status_text(status: str) -> str:
    return _STATUS_TEXT.get(status, _STATUS_TEXT["blocked"])


@dataclass(frozen=True)
class StatusVoiceResult:
    status: Literal["spoken", "synthesized-only", "unavailable"]
    diagnostic: str = ""
```

Add `deliver_agent_status(result, voice_adapter, playback_adapter)`:

- synthesize only `status_text(result.status)`;
- if synthesis unavailable return `StatusVoiceResult("unavailable", ...)`;
- if synthesis succeeds but playback unavailable return `StatusVoiceResult("synthesized-only", ...)`;
- if playback succeeds return `StatusVoiceResult("spoken")`.

- [ ] **Step 4: Add tests proving voice failures cannot alter the agent result**

Use an existing `AgentResult(status="passed", attempts=1)` and fakes:

```python
def test_delivery_failure_is_reported_separately_from_agent_result() -> None:
    agent = AgentResult(status="passed", attempts=1)
    delivery = deliver_agent_status(agent, FailedVoice(), FakePlayback())
    assert agent.status == "passed"
    assert delivery.status == "unavailable"
```

Also cover blocked results and synthesized-only playback failures.

- [ ] **Step 5: Modify the coding CLI parser without changing default behavior**

Update `src/distilling_agents/cli.py` so `main()` accepts optional `argv` for tests:

```python
def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
```

Add these arguments:

```text
--voice / --no-voice        argparse.BooleanOptionalAction; default from DISTILL_VOICE, otherwise False
--voice-language            default DISTILL_VOICE_LANGUAGE or pst
--omnivoice-command         default DISTILL_OMNIVOICE_COMMAND or omnivoice-infer
--voice-output-dir          default ~/.cache/distilling-agents/voice
--voice-player              optional player executable
--voice-ref-audio           optional Path
--voice-ref-text            optional str
```

Boolean env parsing must accept only `1`, `true`, `yes`, `on` as true.

The default remains voice disabled so existing installs and benchmark workflows are unaffected.

- [ ] **Step 6: Insert notification after the coding result is finalized**

Preserve this order:

```text
run_agent()
leave temporary worktree
print full AgentResult JSON
if --voice:
    synthesize deterministic Pashto status
    attempt local playback
return 0 for passed, 2 for blocked
```

Do not run TTS inside the worktree context and do not pass the diff/error log to TTS.

Implementation shape:

```python
print(result.model_dump_json(indent=2))

if args.voice:
    config = VoiceConfig(
        enabled=True,
        language=args.voice_language,
        omnivoice_command=args.omnivoice_command,
        output_dir=args.voice_output_dir,
        ref_audio=args.voice_ref_audio,
        ref_text=args.voice_ref_text,
    )
    delivery = deliver_agent_status(
        result,
        OmniVoiceAdapter(config),
        PlaybackAdapter(player=args.voice_player),
    )
    if delivery.status != "spoken":
        print(f"Voice: {delivery.status} — {delivery.diagnostic}", file=sys.stderr)

return 0 if result.status == "passed" else 2
```

Import `sys` and the voice modules only in the CLI layer. Do not modify `graph.py`.

- [ ] **Step 7: Write CLI regression tests for default-off and exit-code isolation**

Create `tests/test_cli_voice.py`.

At minimum verify parser defaults:

```python
from distilling_agents.cli import parser


def test_existing_cli_defaults_voice_off() -> None:
    args = parser().parse_args(["/tmp/repo", "--issue", "fix it"])
    assert args.voice is False
    assert args.voice_language == "pst"
```

Then test the notification helper path with monkeypatches/fakes so:

- coding `passed` + voice unavailable returns CLI exit `0`;
- coding `blocked` + voice unavailable returns CLI exit `2`;
- voice-disabled path never constructs `OmniVoiceAdapter`.

Use temporary fake worktree context and monkeypatch `run_agent`; do not start vLLM or OmniVoice.

- [ ] **Step 8: Run all status/CLI tests**

```bash
pytest tests/test_status_voice.py tests/test_cli_voice.py -v
```

Expected: all pass.

- [ ] **Step 9: Run the full existing suite to prove the repair loop remains unaffected**

```bash
pytest
```

Expected: all pre-existing graph/tool/worker tests plus new voice tests pass. No GPU/model/audio dependencies are required.

- [ ] **Step 10: Commit coding CLI voice notifications**

```bash
git add src/distilling_agents/status_voice.py src/distilling_agents/cli.py tests/test_status_voice.py tests/test_cli_voice.py
git commit -m "feat: speak optional Pashto agent status"
```

---

### Task 5: Document isolated setup and perform manual Central Pashto acceptance

**Files:**
- Create: `docs/voice.md`
- Modify: `README.md`

**Interfaces:**
- Documentation only; no new runtime dependency.
- Manual acceptance is the first point at which real OmniVoice/CUDA/audio hardware is required.

- [ ] **Step 1: Write `docs/voice.md` with the exact environment boundary**

Document this recommended layout:

```text
Distilling-Agents-/.venv/          # coding harness only
~/.local/omnivoice-venv/           # OmniVoice + PyTorch/audio dependencies
```

Include Linux/WSL2 setup commands based on the upstream OmniVoice README:

```bash
python3 -m venv ~/.local/omnivoice-venv
source ~/.local/omnivoice-venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install omnivoice==0.2.1
```

State that upstream version/CUDA requirements may change and should be rechecked before future upgrades; these commands match the integration baseline used by this plan.

- [ ] **Step 2: Document standalone Central Pashto usage**

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

Audible playback:

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --play
```

Reference voice:

```bash
distill-speak "کار بشپړ شو" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --ref-audio /path/to/pashto-reference.wav \
  --ref-text "د ريفرنس غږ متن"
```

State explicitly: only use reference audio the user has permission to clone.

- [ ] **Step 3: Document coding-agent notification usage and GPU contention**

Show:

```bash
distill-agent /path/to/repo \
  --issue "fix the bug" \
  --voice \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

Document that on an 8 GB single-GPU machine, a running Qwen/vLLM server may leave insufficient VRAM for OmniVoice. V1 reports voice as unavailable rather than killing/restarting vLLM. For manual voice acceptance, stop vLLM first if CUDA OOM occurs, then retry `distill-speak`.

- [ ] **Step 4: Document WSL2 playback requirements**

Explain that automatic playback looks for `ffplay`, `paplay`, then `aplay` unless `--player`/`--voice-player` is configured. Give one concrete Ubuntu option:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Then verify:

```bash
ffplay -nodisp -autoexit out.wav
```

Do not make audio-player packages Python dependencies.

- [ ] **Step 5: Add a concise README section**

Add a short section after the existing run instructions:

```markdown
## Central Pashto voice

Distilling Agents can optionally synthesize status speech through an isolated OmniVoice installation. Central Pashto (`pst`) is the default language.

```bash
distill-speak "سلام، څنګه یې؟" --play
```

The coding harness does not install or import OmniVoice directly. See `docs/voice.md` for the isolated WSL2/CUDA setup, voice cloning, and GPU-contention notes.
```

Also state that voice remains disabled during the coding benchmark.

- [ ] **Step 6: Run documentation-adjacent verification**

```bash
python -m pip install -e '.[dev]'
distill-speak --help
distill-agent --help
pytest
```

Expected: both commands load without OmniVoice installed; full tests pass.

- [ ] **Step 7: Run the real manual Central Pashto synthesis acceptance on the target machine**

With the dedicated OmniVoice environment installed and vLLM stopped if required:

```bash
distill-speak "سلام، څنګه یې؟ کار بشپړ شو." \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --output /tmp/distilling-agents-pst.wav \
  --play
```

Acceptance requirements:

1. command exits `0`;
2. `/tmp/distilling-agents-pst.wav` exists and is non-empty;
3. audible speech is produced through at least one supported player;
4. the speech is recognizably Central Pashto to the user;
5. no coding-agent process is modified or terminated by the voice layer.

If synthesis succeeds but pronunciation/voice quality is poor, record that as a model-quality finding rather than a harness failure. Do not change the benchmark pass criteria.

- [ ] **Step 8: Run GitHub Actions verification before integration**

Push the branch and require the repository's Python 3.11 `pytest` workflow to pass. CI must not install OmniVoice or download its model.

- [ ] **Step 9: Commit documentation**

```bash
git add docs/voice.md README.md
git commit -m "docs: document Central Pashto voice setup"
```

---

## Final Verification Checklist

Run before requesting merge:

```bash
pytest
python -m pip install -e '.[dev]'
distill-speak --help
distill-agent --help
```

Verify in code review:

- `graph.py` has no voice imports or TTS calls.
- `pyproject.toml` has no OmniVoice/Torch dependency additions.
- every subprocess uses argument lists and `shell=False`/default behavior.
- all process errors become structured voice/playback results.
- coding CLI exit status still depends only on `AgentResult.status`.
- automatic status speech contains no diffs, logs, file paths, SHAs, or test output.
- default language is `pst` everywhere unless explicitly overridden.
- voice defaults off for the coding CLI.
- `distill-speak` works as a standalone WAV generator without `--play`.
- benchmark implementation remains independent from voice.

Then perform the real target-machine acceptance command from Task 5. Do not claim end-to-end Central Pashto audio works until that manual GPU/audio test has actually succeeded.
