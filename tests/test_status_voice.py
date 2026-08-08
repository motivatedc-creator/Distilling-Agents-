from distilling_agents.models import AgentResult
from distilling_agents.playback import PlaybackResult
from distilling_agents.status_voice import deliver_agent_status, status_text
from distilling_agents.voice import VoiceResult


class FailedVoice:
    def synthesize(self, text, output_path=None):
        return VoiceResult("unavailable", diagnostic="oom")


class ExplodingVoice:
    def synthesize(self, text, output_path=None):
        raise RuntimeError("unexpected tts bug")


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


class ExplodingPlayback:
    def play(self, wav_path):
        raise RuntimeError("unexpected playback bug")


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


def test_unexpected_voice_exception_is_contained() -> None:
    delivery = deliver_agent_status(
        AgentResult(status="passed", attempts=1),
        ExplodingVoice(),
        FakePlayback(),
    )
    assert delivery.status == "unavailable"
    assert "unexpected tts bug" in delivery.diagnostic


def test_playback_failure_is_synthesized_only(tmp_path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    delivery = deliver_agent_status(
        AgentResult(status="blocked", attempts=3),
        FakeVoice(wav),
        FakePlayback("unavailable"),
    )
    assert delivery.status == "synthesized-only"


def test_unexpected_playback_exception_is_contained(tmp_path) -> None:
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"RIFF")
    delivery = deliver_agent_status(
        AgentResult(status="passed", attempts=1),
        FakeVoice(wav),
        ExplodingPlayback(),
    )
    assert delivery.status == "synthesized-only"
    assert "unexpected playback bug" in delivery.diagnostic
