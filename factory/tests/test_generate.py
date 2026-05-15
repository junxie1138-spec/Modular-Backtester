import json
from pathlib import Path

import pytest

from factory.generate import (
    GenerationError,
    parse_claude_output,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_parses_clean_output() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_clean.json"))
    assert parsed["strategy_id"] == "gen_1"
    assert parsed["one_line_summary"] == "sma cross 20/100"
    assert parsed["allow_short"] is False
    assert parsed["strategy_file"].startswith("# placeholder")
    assert cost == pytest.approx(0.034)


def test_strips_markdown_fences() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_fenced.json"))
    assert parsed["strategy_id"] == "gen_2"
    assert cost == pytest.approx(0.041)


def test_locates_json_inside_prose() -> None:
    parsed, cost = parse_claude_output(_load("claude_output_prose_wrapped.json"))
    assert parsed["strategy_id"] == "gen_3"
    assert cost == pytest.approx(0.029)


def test_raises_on_no_json_object() -> None:
    with pytest.raises(GenerationError) as exc:
        parse_claude_output(_load("claude_output_malformed.json"))
    assert "no JSON object" in str(exc.value).lower() or "could not parse" in str(exc.value).lower()


def test_raises_on_broken_envelope() -> None:
    with pytest.raises(GenerationError):
        parse_claude_output("this is not even JSON")


def test_raises_on_missing_required_keys() -> None:
    # Envelope is fine; inner JSON is parseable but lacks required keys.
    envelope = json.dumps({
        "result": '{"strategy_id": "gen_x"}',
        "total_cost_usd": 0.01,
    })
    with pytest.raises(GenerationError) as exc:
        parse_claude_output(envelope)
    assert "missing" in str(exc.value).lower()


def test_raises_on_envelope_without_result_field() -> None:
    envelope = json.dumps({"session_id": "x", "total_cost_usd": 0.0})
    with pytest.raises(GenerationError):
        parse_claude_output(envelope)
