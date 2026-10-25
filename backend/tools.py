"""Gemini tool declarations for the surgical tray log.

``FUNCTION_DECLARATIONS`` is the JSON function-declaration list a Gemini Live
session consumes (as ``tools[0]["function_declarations"]``). The same objects
are registered in ``shared_schema.json`` under ``x-gemini.tools`` so the
bridge accepts ``log_instrument`` and ``verify_tray_count`` frames and
``GET /api/gemini/tools`` serves them.

``log_instrument`` and ``verify_tray_count`` are the Python callables those
declarations bind to. ``dispatch`` runs one by name and returns a JSON object
suitable for a ``function_result`` frame.

Count rules:
* ``added`` increases the on-tray count.
* ``removed`` decreases it (the count may go negative when more were removed
  than added; that mismatch is the signal a count is off).
* ``used`` is written to the log and does not change the on-tray count.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ACTIONS = ("added", "removed", "used")

LOG_INSTRUMENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instrument": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "Common name of the instrument, for example scalpel, forceps, needle driver, or sponge.",
        },
        "action": {
            "type": "string",
            "enum": list(ACTIONS),
            "description": "added puts instruments on the tray, removed takes them off, and used records use without changing the on-tray count.",
        },
        "quantity": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "How many of that instrument this entry accounts for. Defaults to 1 when omitted.",
        },
        "notes": {
            "type": "string",
            "maxLength": 280,
            "description": "Optional detail such as size or laterality. Omit when the user gave none.",
        },
    },
    "required": ["instrument", "action"],
}

VERIFY_TRAY_COUNT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expected_count": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "description": "How many the team expects to be on the tray right now.",
        },
        "instrument": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "Instrument to count. Omit, or leave empty, to verify the whole tray as one total.",
        },
    },
    "required": ["expected_count"],
}

LOG_INSTRUMENT_DECLARATION: dict[str, Any] = {
    "name": "log_instrument",
    "description": "Record a surgical instrument on the case log. Call this when the user adds an instrument to the tray, removes one, or notes that one was used. added and removed change the running tray count; used does not.",
    "parameters": LOG_INSTRUMENT_PARAMETERS,
}

VERIFY_TRAY_COUNT_DECLARATION: dict[str, Any] = {
    "name": "verify_tray_count",
    "description": "Compare the running tray count with the count the team expects. Call this when the user asks to verify, confirm, or reconcile the tray or one instrument. The actual count comes from earlier log_instrument calls. The result reports actual_count, matches, and delta (actual minus expected).",
    "parameters": VERIFY_TRAY_COUNT_PARAMETERS,
}

# Flat list, same shape as shared_schema.json x-gemini.tools entries and as
# GET /api/gemini/tools -> tools[].
FUNCTION_DECLARATIONS: list[dict[str, Any]] = [
    LOG_INSTRUMENT_DECLARATION,
    VERIFY_TRAY_COUNT_DECLARATION,
]

# Session payload for the Gemini Live API: config["tools"] = GEMINI_TOOLS.
GEMINI_TOOLS: list[dict[str, Any]] = [
    {"function_declarations": FUNCTION_DECLARATIONS},
]

TOOL_NAMES = frozenset(declaration["name"] for declaration in FUNCTION_DECLARATIONS)


class ToolError(Exception):
    """The tool rejected the call. ``code`` and ``message`` fit a function_result error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now_ms() -> int:
    return int(time.time() * 1000)


def _as_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ToolError("invalid_args", f"{name} must be an integer")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ToolError("invalid_args", f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ToolError("invalid_args", f"{name} must be between {minimum} and {maximum}")
    return value


def _instrument_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ToolError("invalid_args", "instrument must be a string")
    name = " ".join(value.split())
    if not name:
        raise ToolError("invalid_args", "instrument must not be empty")
    if len(name) > 80:
        raise ToolError("invalid_args", "instrument must be 80 characters or fewer")
    return name


def _optional_instrument(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _instrument_name(value)


def _action(value: Any) -> str:
    if not isinstance(value, str):
        raise ToolError("invalid_args", "action must be a string")
    action = value.strip().casefold()
    if action not in ACTIONS:
        allowed = ", ".join(ACTIONS)
        raise ToolError("invalid_args", f"action must be one of {allowed}")
    return action


def _notes(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError("invalid_args", "notes must be a string")
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) > 280:
        raise ToolError("invalid_args", "notes must be 280 characters or fewer")
    return text


@dataclass(frozen=True, slots=True)
class _Entry:
    instrument: str
    action: str
    quantity: int
    notes: str | None
    recorded_at_ms: int


def _labeled_counts(entries: list[_Entry]) -> dict[str, int]:
    """Net on-tray count per instrument, keyed by the latest spelling of its name."""
    totals: dict[str, int] = {}
    labels: dict[str, str] = {}
    for entry in entries:
        key = entry.instrument.casefold()
        labels[key] = entry.instrument
        if entry.action == "added":
            totals[key] = totals.get(key, 0) + entry.quantity
        elif entry.action == "removed":
            totals[key] = totals.get(key, 0) - entry.quantity
        else:
            totals.setdefault(key, 0)
    return {labels[key]: totals[key] for key in totals}


def _count_named(labeled: dict[str, int], instrument: str) -> int:
    folded = instrument.casefold()
    for label, count in labeled.items():
        if label.casefold() == folded:
            return count
    return 0


class TrayLog:
    """In-memory case log shared by the tool functions."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def record(
        self,
        instrument: str,
        action: str,
        quantity: int,
        notes: str | None,
    ) -> dict[str, Any]:
        entry = _Entry(instrument, action, quantity, notes, _now_ms())
        with self._lock:
            self._entries.append(entry)
            labeled = _labeled_counts(self._entries)
        return {
            "instrument": instrument,
            "action": action,
            "quantity": quantity,
            "notes": notes,
            "on_tray": _count_named(labeled, instrument),
            "tray_total": sum(labeled.values()),
            "recorded_at_ms": entry.recorded_at_ms,
        }

    def verify(self, expected_count: int, instrument: str | None) -> dict[str, Any]:
        with self._lock:
            labeled = _labeled_counts(self._entries)
        actual = sum(labeled.values()) if instrument is None else _count_named(labeled, instrument)
        return {
            "instrument": instrument,
            "expected_count": expected_count,
            "actual_count": actual,
            "tray_total": sum(labeled.values()),
            "matches": actual == expected_count,
            "delta": actual - expected_count,
            "counts_by_instrument": labeled,
        }


_tray = TrayLog()


def clear_tray() -> None:
    """Drop every logged instrument."""
    _tray.clear()


def log_instrument(
    instrument: str,
    action: str,
    quantity: int = 1,
    notes: str | None = None,
) -> dict[str, Any]:
    """Append one tray-log entry and return the updated counts.

    ``action`` is ``added``, ``removed``, or ``used``. ``quantity`` defaults
    to 1. ``used`` is recorded and leaves the on-tray count unchanged.
    """
    name = _instrument_name(instrument)
    verb = _action(action)
    count = _as_int(quantity, "quantity", minimum=1, maximum=100)
    detail = _notes(notes)
    return _tray.record(name, verb, count, detail)


def verify_tray_count(expected_count: int, instrument: str | None = None) -> dict[str, Any]:
    """Compare ``expected_count`` with the count accumulated from ``log_instrument``.

    Omit ``instrument`` to verify the whole tray. ``delta`` is actual minus
    expected: positive means the tray holds more than the team expects.
    """
    expected = _as_int(expected_count, "expected_count", minimum=0, maximum=1000)
    name = _optional_instrument(instrument)
    return _tray.verify(expected, name)


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "log_instrument": log_instrument,
    "verify_tray_count": verify_tray_count,
}


def dispatch(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a declared tool. Raises :class:`ToolError` for an unknown name or bad arguments."""
    handler = _HANDLERS.get(name)
    if handler is None:
        allowed = ", ".join(sorted(_HANDLERS))
        raise ToolError("unknown_tool", f"unknown tool {name!r}; expected one of {allowed}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("invalid_args", "arguments must be a JSON object")

    params = inspect.signature(handler).parameters
    unknown = sorted(set(args) - set(params))
    if unknown:
        raise ToolError("invalid_args", "unexpected arguments: " + ", ".join(unknown))
    missing = [
        param_name
        for param_name, param in params.items()
        if param.default is inspect.Parameter.empty and param_name not in args
    ]
    if missing:
        raise ToolError("invalid_args", "missing arguments: " + ", ".join(missing))
    bound = {key: args[key] for key in params if key in args}
    return handler(**bound)


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_ACTION_WORDS = {
    "add": "added",
    "added": "added",
    "adding": "added",
    "remove": "removed",
    "removed": "removed",
    "removing": "removed",
    "take": "removed",
    "took": "removed",
    "use": "used",
    "used": "used",
    "using": "used",
}
_VERIFY_WORDS = {"verify", "confirm", "reconcile", "check"}
_VERIFY_SKIP = {"tray", "count", "the", "total", "instruments", "instrument", "of"}
_FILLER = {"please", "um", "uh"}
_CLAUSE_SPLIT = re.compile(r"\s+and\s+", re.IGNORECASE)


def _number_token(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    return _NUMBER_WORDS.get(word)


def _parse_verify_clause(tokens: list[str]) -> tuple[str, dict[str, Any]] | None:
    numbers: list[int] = []
    name: list[str] = []
    for token in tokens:
        word = token.strip(".,").casefold()
        raw = token.strip(".,")
        if not word or word in _VERIFY_SKIP:
            continue
        number = _number_token(word)
        if number is not None:
            numbers.append(number)
            continue
        name.append(raw)
    if not numbers:
        return None
    args: dict[str, Any] = {"expected_count": numbers[-1]}
    if name:
        args["instrument"] = " ".join(name)
    return ("verify_tray_count", args)


def _parse_log_clause(tokens: list[str]) -> tuple[str, dict[str, Any]] | None:
    action = "added"
    quantity: int | None = None
    name: list[str] = []
    for token in tokens:
        word = token.strip(".,").casefold()
        raw = token.strip(".,")
        if not word or (word in _FILLER and not name):
            continue
        if word in _ACTION_WORDS and not name:
            action = _ACTION_WORDS[word]
            continue
        number = _number_token(word)
        if number is not None and quantity is None and not name:
            quantity = number
            continue
        name.append(raw)
    instrument = " ".join(name).strip()
    if not instrument:
        return None
    return (
        "log_instrument",
        {"instrument": instrument, "action": action, "quantity": quantity or 1},
    )


def interpret_log_command(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Turn a wake-word remainder into tray tool calls.

    ``"add sponge and two clamps"`` becomes two ``log_instrument`` calls.
    ``"verify 12"`` becomes ``verify_tray_count``. Unrecognized text is ignored.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    calls: list[tuple[str, dict[str, Any]]] = []
    for clause in _CLAUSE_SPLIT.split(cleaned):
        tokens = clause.strip(" ,.;").split()
        if not tokens:
            continue
        head = tokens[0].strip(".,").casefold()
        parsed = (
            _parse_verify_clause(tokens[1:])
            if head in _VERIFY_WORDS
            else _parse_log_clause(tokens)
        )
        if parsed is not None:
            calls.append(parsed)
    return calls


def live_function_declarations() -> list[dict[str, Any]]:
    """Declarations shaped for a Gemini Live setup message.

    Constraint keywords such as ``minimum`` stay in the Python validators.
    Live setup uses the smaller declaration subset the session accepts, and
    blocking behavior so the model waits for the tool result.
    """
    declarations: list[dict[str, Any]] = []
    for source in (LOG_INSTRUMENT_DECLARATION, VERIFY_TRAY_COUNT_DECLARATION):
        declaration = json.loads(json.dumps(source))
        _strip_schema_constraints(declaration.get("parameters"))
        declaration["behavior"] = "BLOCKING"
        declarations.append(declaration)
    return declarations


def _strip_schema_constraints(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"):
        schema.pop(key, None)
    for value in schema.values():
        if isinstance(value, dict):
            _strip_schema_constraints(value)
        elif isinstance(value, list):
            for item in value:
                _strip_schema_constraints(item)


def _check_shared_schema() -> None:
    """Fail import if the bridge schema drifted from these declarations."""
    path = Path(__file__).with_name("shared_schema.json")
    if not path.is_file():
        return
    declared = {
        tool["name"]: tool
        for tool in json.loads(path.read_text(encoding="utf-8"))["x-gemini"]["tools"]
        if isinstance(tool, dict) and "name" in tool
    }
    for tool in FUNCTION_DECLARATIONS:
        if declared.get(tool["name"]) != tool:
            raise RuntimeError(
                f"{tool['name']} in tools.py does not match shared_schema.json x-gemini.tools"
            )


_check_shared_schema()
