"""End-word command detection and buffering for stable microphone transcripts."""

from __future__ import annotations

import os
import re
from typing import TypedDict

END_WORD = "over"


def _duration_setting(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be zero (disabled) or greater")
    return value


SILENCE_TIMEOUT_MS = _duration_setting("END_WORD_SILENCE_TIMEOUT_MS", 8000)
END_WORD_PAUSE_MS = _duration_setting("END_WORD_PAUSE_MS", 400)
DEBOUNCE_MS = _duration_setting("END_WORD_DEBOUNCE_MS", 1500)
_TRAILING_PUNCTUATION = " \t,.!?;:-\"'"
_END_WORD_PATTERN = re.compile(rf"(?<!\w){re.escape(END_WORD)}(?!\w)[^\w]*$", re.IGNORECASE)


class CommandExtraction(TypedDict):
    command: str
    ended: bool


def extract_command(transcript: str) -> CommandExtraction:
    """Remove a terminal, whole-word "over" and report whether it was present."""
    text = transcript or ""
    match = _END_WORD_PATTERN.search(text)
    if match is None:
        return {"command": text.strip(), "ended": False}
    command = text[: match.start()].rstrip(_TRAILING_PUNCTUATION)
    return {"command": command, "ended": True}


def _event(phase: str, text: str = "") -> dict[str, str]:
    return {"type": "speech_log", "phase": phase, "text": text}


class EndWordBuffer:
    """Accumulate stable phrases until the end word or a discard timeout."""

    def __init__(
        self,
        silence_timeout_ms: int = SILENCE_TIMEOUT_MS,
        end_word_pause_ms: int = END_WORD_PAUSE_MS,
        debounce_ms: int = DEBOUNCE_MS,
    ) -> None:
        self.silence_timeout_ms = silence_timeout_ms
        self.end_word_pause_ms = end_word_pause_ms
        self.debounce_ms = debounce_ms
        self.active = False
        self.text = ""
        self.last_activity_ms: float | None = None
        self.end_word_at_ms: float | None = None
        self.last_command: str | None = None
        self.last_command_at_ms: float | None = None

    def start(self, now_ms: float) -> list[dict[str, str]]:
        self.active = True
        self.text = ""
        self.last_activity_ms = now_ms
        self.end_word_at_ms = None
        return [_event("recording")]

    def append(self, transcript: str, now_ms: float) -> list[dict[str, str]]:
        if not self.active:
            return []
        phrase = transcript.strip()
        if not phrase:
            return []
        if (
            self.end_word_at_ms is not None
            and extract_command(phrase) == {"command": "", "ended": True}
        ):
            return []
        self.text = f"{self.text} {phrase}".strip()
        self.last_activity_ms = now_ms
        extracted = extract_command(self.text)
        self.end_word_at_ms = now_ms if extracted["ended"] else None
        return [_event("recording", extracted["command"] if extracted["ended"] else self.text)]

    def poll(self, now_ms: float) -> list[dict[str, str]]:
        if not self.active:
            return []
        if (
            self.end_word_at_ms is not None
            and now_ms - self.end_word_at_ms >= self.end_word_pause_ms
        ):
            extracted = extract_command(self.text)
            command = extracted["command"]
            self._reset()
            if not command:
                return [_event("idle")]
            normalized = " ".join(command.casefold().split())
            if (
                self.last_command == normalized
                and self.last_command_at_ms is not None
                and now_ms - self.last_command_at_ms < self.debounce_ms
            ):
                return [_event("idle")]
            self.last_command = normalized
            self.last_command_at_ms = now_ms
            return [_event("final", command), _event("idle")]

        if (
            self.silence_timeout_ms > 0
            and self.last_activity_ms is not None
            and now_ms - self.last_activity_ms >= self.silence_timeout_ms
        ):
            self._reset()
            return [_event("idle")]
        return []

    def _reset(self) -> None:
        self.active = False
        self.text = ""
        self.last_activity_ms = None
        self.end_word_at_ms = None
