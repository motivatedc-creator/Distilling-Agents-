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
