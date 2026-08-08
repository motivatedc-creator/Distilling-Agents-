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
