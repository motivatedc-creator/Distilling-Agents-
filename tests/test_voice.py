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
        return subprocess.CompletedProcess(
            argv,
            self.returncode,
            "ok",
            "boom" if self.returncode else "",
        )


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


def test_stale_existing_output_is_not_accepted_as_fresh_synthesis(tmp_path: Path) -> None:
    output = tmp_path / "out.wav"
    output.write_bytes(b"RIFFold-audio")
    runner = FakeRunner(write_output=False)

    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=runner).synthesize(
        "سلام",
        output_path=output,
    )

    assert result.status == "unavailable"
    assert not output.exists()


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


def test_diagnostic_is_bounded(tmp_path: Path) -> None:
    class LoudRunner(FakeRunner):
        def __call__(self, args, **kwargs):
            return subprocess.CompletedProcess(list(args), 1, "x" * 8000, "TAIL")

    result = OmniVoiceAdapter(VoiceConfig(output_dir=tmp_path), runner=LoudRunner()).synthesize("سلام")
    assert len(result.diagnostic) <= 4000
    assert "TAIL" in result.diagnostic
