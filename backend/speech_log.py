"""Pure function extracting the speech-log remainder after a wake word.

Matches the wake-word logic in voice.wake_remainder (whole-word, case-insensitive,
\b boundaries, same trailing punctuation strip set) plus the armed/recording
phase rules for the surgical log.
"""

import re

_TRAILING_PUNCTUATION = " \t,.!?;:-\"'"


def split_trailing_exit(text: str) -> tuple[str, bool]:
    """Drop a trailing whole-word ``exit``. The flag is true when it was spoken."""
    stripped = (text or "").strip().strip(_TRAILING_PUNCTUATION)
    if not stripped:
        return "", False
    if re.fullmatch(r"exit", stripped, flags=re.IGNORECASE):
        return "", True
    match = re.fullmatch(r"(.*\S)\s+exit", stripped, flags=re.IGNORECASE)
    if match is None:
        return stripped, False
    return match.group(1).strip().strip(_TRAILING_PUNCTUATION), True


def speech_log_from_transcript(
    transcript: str, wake_word: str = "log", *, armed: bool = False
) -> dict[str, str] | None:
    """Return {"type": "speech_log", "phase": "...", "text": "..."} or None.

    See module docstring and test expectations for full contract.
    """
    if not wake_word or not wake_word.strip():
        return None
    wake = wake_word.strip()

    transcript = transcript or ""
    match = re.search(rf"\b{re.escape(wake)}\b", transcript, flags=re.IGNORECASE)
    if match is None:
        remainder: str | None = None
    else:
        remainder = transcript[match.end() :].strip(_TRAILING_PUNCTUATION)

    if not armed:
        if match is None:
            return None
        if remainder == "":
            return {"type": "speech_log", "phase": "recording", "text": ""}
        return {"type": "speech_log", "phase": "final", "text": remainder}

    # armed=True
    if not transcript.strip():
        return None
    if match is not None:
        if remainder == "":
            return {"type": "speech_log", "phase": "recording", "text": ""}
        return {"type": "speech_log", "phase": "final", "text": remainder}
    # wake absent when armed: final with stripped transcript, or None if empty
    text = transcript.strip().strip(_TRAILING_PUNCTUATION)
    if not text:
        return None
    return {"type": "speech_log", "phase": "final", "text": text}


class SpeechLogSession:
    def __init__(self, wake_word: str = "log", silence_s: float = 1.5, arm_hold_s: float = 6.0) -> None:
        self.phase = "idle"
        self.text = ""
        self._wake_word = (wake_word or "log").strip() or "log"
        self._silence_s = float(silence_s)
        self._arm_hold_s = float(arm_hold_s)
        self._gap = 0.0

    def arm(self) -> list[dict[str, str]]:
        self._gap = 0.0
        if self.phase != "recording":
            self.phase = "recording"
            self.text = ""
        return [{"type": "speech_log", "phase": "recording", "text": self.text}]

    def hear(self, transcript: str) -> list[dict[str, str]]:
        self._gap = 0.0
        transcript = transcript or ""
        if not self._wake_word:
            return []
        wake = self._wake_word
        match = re.search(rf"\b{re.escape(wake)}\b", transcript, flags=re.IGNORECASE)
        if match is None:
            rem = None
        else:
            rem = transcript[match.end() :].strip(_TRAILING_PUNCTUATION)
        if self.phase == "idle":
            if rem is None:
                return []
            addition, stopped = split_trailing_exit(rem)
            if stopped and not addition:
                return []
            self.phase = "recording"
            self.text = addition
            events = [{"type": "speech_log", "phase": "recording", "text": self.text}]
            if stopped:
                events.extend(self._finish())
            return events
        if self.phase == "recording":
            if rem is not None and rem == "":
                return [{"type": "speech_log", "phase": "recording", "text": self.text}]
            source = rem if rem is not None else transcript.strip().strip(_TRAILING_PUNCTUATION)
            addition, stopped = split_trailing_exit(source)
            if addition:
                self.text = f"{self.text} {addition}".strip() if self.text else addition
            if stopped:
                return self._finish()
            return [{"type": "speech_log", "phase": "recording", "text": self.text}]
        return []

    def _finish(self) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        if self.text:
            events.append({"type": "speech_log", "phase": "final", "text": self.text})
            events.append({"type": "speech_log", "phase": "idle", "text": ""})
        else:
            events.append({"type": "speech_log", "phase": "idle", "text": ""})
        self.phase = "idle"
        self.text = ""
        self._gap = 0.0
        return events

    def gap(self, seconds: float) -> list[dict[str, str]]:
        if self.phase != "recording" or seconds <= 0:
            return []
        self._gap += seconds
        threshold = self._arm_hold_s if not self.text else self._silence_s
        if self._gap < threshold:
            return []
        return self._finish()

