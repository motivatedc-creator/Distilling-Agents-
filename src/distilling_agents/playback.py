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
                "unavailable",
                "WAV file is missing or empty",
            )
        player = self._resolve_player()
        if player is None:
            return PlaybackResult(
                "unavailable",
                "No supported local audio player found",
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
                "unavailable",
                f"Audio player not found: {player}",
            )
        except OSError as exc:
            return PlaybackResult(
                "unavailable",
                _tail(f"Playback process error: {exc}"),
            )

        diagnostic = _tail(
            (completed.stdout + "\n" + completed.stderr).strip()
        )
        if completed.returncode != 0:
            return PlaybackResult("unavailable", diagnostic)
        return PlaybackResult("played", diagnostic)
