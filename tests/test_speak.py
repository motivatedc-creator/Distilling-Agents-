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
