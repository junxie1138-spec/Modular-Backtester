from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_KEYS: tuple[str, ...] = (
    "strategy_id",
    "one_line_summary",
    "hypothesis",
    "novelty_justification",
    "failure_mode",
    "allow_short",
    "strategy_file",
    "config_file",
)


class GenerationError(RuntimeError):
    """Raised when claude -p output cannot be parsed into the expected shape."""


@dataclass(slots=True, frozen=True)
class GenerationResult:
    parsed: dict[str, Any]
    cost_usd: float
    raw_stdout: str


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1)
    return text


def _find_outer_json_object(text: str) -> str:
    """Locate the outermost balanced {...} in text. Raises ValueError if none."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in text")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces in text")


def parse_claude_output(stdout: str) -> tuple[dict[str, Any], float]:
    """Defensive double-unwrap.

    Returns (parsed_strategy_dict, total_cost_usd).
    Raises GenerationError on any parse failure or missing required key.
    """
    # Layer 1: CLI envelope.
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"could not parse CLI envelope: {exc}") from exc
    if not isinstance(envelope, dict) or "result" not in envelope:
        raise GenerationError("envelope missing 'result' field")
    inner_text = envelope["result"]
    cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)

    # Layer 2: strip fences, locate outer JSON, parse.
    stripped = _strip_fences(inner_text)
    try:
        inner_blob = _find_outer_json_object(stripped)
    except ValueError as exc:
        raise GenerationError(f"could not parse inner result — no JSON object found: {exc}") from exc
    try:
        parsed = json.loads(inner_blob)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"could not parse inner JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GenerationError("inner JSON is not an object")

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        raise GenerationError(f"inner JSON missing keys: {missing}")

    return parsed, cost


def call_claude(
    *,
    prompt: str,
    claude_cmd: str,
    claude_flags: tuple[str, ...],
    timeout_sec: int,
) -> GenerationResult:
    """Invoke claude -p as a subprocess and parse its stdout.

    Raises GenerationError on non-zero exit, timeout, or unparseable output.
    """
    cmd = [claude_cmd, *claude_flags, prompt]
    log.info("calling claude (cmd=%s flags=%s)", claude_cmd, claude_flags)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(f"claude -p timed out after {timeout_sec}s") from exc
    except FileNotFoundError as exc:
        raise GenerationError(f"claude command not found: {claude_cmd}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-500:]
        raise GenerationError(
            f"claude -p exited {proc.returncode}; stderr tail: {tail}"
        )

    parsed, cost = parse_claude_output(proc.stdout)
    return GenerationResult(parsed=parsed, cost_usd=cost, raw_stdout=proc.stdout)
