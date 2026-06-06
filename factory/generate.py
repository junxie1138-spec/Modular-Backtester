from __future__ import annotations

import json
import logging
import re
import shutil
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
    """Raised when generator CLI output cannot be parsed into the expected shape."""


@dataclass(slots=True, frozen=True)
class GenerationResult:
    parsed: dict[str, Any]
    cost_usd: float
    raw_stdout: str
    usage: dict[str, int] | None = None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

# Maps each normalized token key -> the raw field name in the CLI envelope's
# `usage` object. Verified against a live `claude -p --output-format json`
# invocation (see the token-tracking plan, Task 1 Step 1).
_USAGE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_creation", "cache_creation_input_tokens"),
    ("cache_read", "cache_read_input_tokens"),
)


def _extract_usage(envelope: dict[str, Any]) -> dict[str, int] | None:
    """Normalize the CLI envelope's `usage` block into a flat token dict.

    Returns None when the envelope carries no `usage` object at all. When
    `usage` is present, every component absent from it is treated as 0.
    """
    raw = envelope.get("usage")
    if not isinstance(raw, dict):
        return None
    return {norm: int(raw.get(cli) or 0) for norm, cli in _USAGE_FIELD_MAP}


def _validate_strategy_payload(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise GenerationError("inner JSON is not an object")

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing:
        raise GenerationError(f"inner JSON missing keys: {missing}")
    return parsed


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


def _find_envelope_with_result(stdout: str) -> dict[str, Any]:
    """Scan stdout for any top-level JSON object that has a 'result' field.

    Tolerates: leading/trailing whitespace, prefix or suffix non-JSON text,
    multiple JSON objects on stdout (e.g. when claude emits diagnostic JSON
    before/after the result envelope without --bare), and pretty-printed
    multi-line JSON. Returns the first matching object.
    """
    pos = 0
    while pos < len(stdout):
        start = stdout.find("{", pos)
        if start < 0:
            break
        try:
            obj_str = _find_outer_json_object(stdout[start:])
            obj = json.loads(obj_str)
        except (ValueError, json.JSONDecodeError):
            pos = start + 1
            continue
        if isinstance(obj, dict) and "result" in obj:
            return obj
        pos = start + len(obj_str)
    raise GenerationError(
        "could not parse CLI envelope: no JSON object with 'result' field "
        "found in stdout"
    )


def parse_claude_output(stdout: str) -> tuple[dict[str, Any], float, dict[str, int] | None]:
    """Defensive double-unwrap.

    Returns (parsed_strategy_dict, total_cost_usd, usage_token_dict_or_None).
    Raises GenerationError on any parse failure or missing required key.
    """
    # Layer 1: CLI envelope. We scan for a {...} block with a 'result' field
    # rather than parsing the entire stdout, because claude without --bare can
    # emit additional diagnostic content on stdout (hooks output, plugin sync,
    # CLAUDE.md auto-discovery, etc.) that breaks a strict json.loads(stdout).
    envelope = _find_envelope_with_result(stdout)
    inner_text = envelope["result"]
    cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)
    usage = _extract_usage(envelope)

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
    return _validate_strategy_payload(parsed), cost, usage


def parse_codex_output(stdout: str) -> tuple[dict[str, Any], float, dict[str, int] | None]:
    """Parse Codex CLI final-message stdout into the factory payload shape.

    `codex exec -` prints the final agent message to stdout. The prompt asks for
    strict JSON, but this parser tolerates fences or surrounding prose for the
    same reason the Claude parser does: a bad wrapper should not waste a whole
    cycle when the inner strategy payload is valid.
    """
    stripped = _strip_fences(stdout.strip())
    try:
        inner_blob = _find_outer_json_object(stripped)
    except ValueError as exc:
        raise GenerationError(f"could not parse codex output — no JSON object found: {exc}") from exc
    try:
        parsed = json.loads(inner_blob)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"could not parse codex JSON: {exc}") from exc
    return _validate_strategy_payload(parsed), 0.0, None


def parse_generator_output(
    stdout: str,
    *,
    provider: str,
) -> tuple[dict[str, Any], float, dict[str, int] | None]:
    provider_norm = provider.lower()
    if provider_norm == "claude":
        return parse_claude_output(stdout)
    if provider_norm in {"codex", "gpt", "openai"}:
        return parse_codex_output(stdout)
    raise GenerationError(
        f"unsupported generation provider {provider!r}; expected 'claude' or 'codex'"
    )


def _resolve_cmd(cmd: str, *, provider: str) -> str:
    resolved = shutil.which(cmd)
    if resolved is None:
        raise GenerationError(f"{provider} command not found on PATH: {cmd}")
    return resolved


def call_generator(
    *,
    prompt: str,
    provider: str,
    cmd: str,
    flags: tuple[str, ...],
    timeout_sec: int,
) -> GenerationResult:
    """Invoke the configured generation provider and parse stdout.

    The prompt is piped through stdin for both supported providers. Claude needs
    this because `--allowedTools` is variadic; Codex supports `codex exec -` to
    read the full prompt from stdin.
    """
    provider_norm = provider.lower()
    if provider_norm not in {"claude", "codex", "gpt", "openai"}:
        raise GenerationError(
            f"unsupported generation provider {provider!r}; expected 'claude' or 'codex'"
        )
    resolved = _resolve_cmd(cmd, provider=provider_norm)
    argv = [resolved, *flags]
    log.info("calling generator provider=%s resolved=%s flags=%s", provider_norm, resolved, flags)
    try:
        proc = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(f"{provider_norm} generation timed out after {timeout_sec}s") from exc
    except FileNotFoundError as exc:
        raise GenerationError(f"{provider_norm} command not found: {cmd}") from exc
    except OSError as exc:
        raise GenerationError(f"{provider_norm} subprocess failed to start: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-500:]
        raise GenerationError(
            f"{provider_norm} generation exited {proc.returncode}; stderr tail: {tail}"
        )

    parsed, cost, usage = parse_generator_output(proc.stdout, provider=provider_norm)
    return GenerationResult(parsed=parsed, cost_usd=cost, raw_stdout=proc.stdout, usage=usage)


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
    return call_generator(
        prompt=prompt,
        provider="claude",
        cmd=claude_cmd,
        flags=claude_flags,
        timeout_sec=timeout_sec,
    )
