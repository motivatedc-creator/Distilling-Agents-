from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import distilling_agents.cli as cli
from distilling_agents.models import AgentResult
from distilling_agents.voice import VoiceResult


class DummyWorker:
    def __init__(self, **kwargs):
        pass


class FailedAdapter:
    def __init__(self, config):
        pass

    def synthesize(self, text, output_path=None):
        return VoiceResult("unavailable", diagnostic="oom")


class DummyPlayback:
    def __init__(self, **kwargs):
        pass

    def play(self, wav_path):
        raise AssertionError("playback must not run when synthesis fails")


@contextmanager
def fake_worktree(repo: Path):
    yield repo


def test_existing_cli_defaults_voice_off() -> None:
    args = cli.parser().parse_args(["/tmp/repo", "--issue", "fix it"])
    assert args.voice is False
    assert args.voice_language == "pst"


def test_voice_can_be_enabled_explicitly() -> None:
    args = cli.parser().parse_args([
        "/tmp/repo",
        "--issue",
        "fix it",
        "--voice",
    ])
    assert args.voice is True


def test_voice_failure_does_not_change_passed_exit(monkeypatch) -> None:
    monkeypatch.setattr(cli, "VLLMWorker", DummyWorker)
    monkeypatch.setattr(cli, "temporary_worktree", fake_worktree)
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda **kwargs: AgentResult(status="passed", attempts=1),
    )
    monkeypatch.setattr(cli, "OmniVoiceAdapter", FailedAdapter)
    monkeypatch.setattr(cli, "PlaybackAdapter", DummyPlayback)

    assert cli.main(["/tmp/repo", "--issue", "fix it", "--voice"]) == 0


def test_voice_failure_does_not_change_blocked_exit(monkeypatch) -> None:
    monkeypatch.setattr(cli, "VLLMWorker", DummyWorker)
    monkeypatch.setattr(cli, "temporary_worktree", fake_worktree)
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda **kwargs: AgentResult(status="blocked", attempts=3),
    )
    monkeypatch.setattr(cli, "OmniVoiceAdapter", FailedAdapter)
    monkeypatch.setattr(cli, "PlaybackAdapter", DummyPlayback)

    assert cli.main(["/tmp/repo", "--issue", "fix it", "--voice"]) == 2
