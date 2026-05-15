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
    parsed, cost, usage = parse_claude_output(_load("claude_output_clean.json"))
    assert parsed["strategy_id"] == "gen_1"
    assert parsed["one_line_summary"] == "sma cross 20/100"
    assert parsed["allow_short"] is False
    assert parsed["strategy_file"].startswith("# placeholder")
    assert cost == pytest.approx(0.034)
    # The clean fixture has no `usage` block -> usage is None.
    assert usage is None


def test_strips_markdown_fences() -> None:
    parsed, cost, usage = parse_claude_output(_load("claude_output_fenced.json"))
    assert parsed["strategy_id"] == "gen_2"
    assert cost == pytest.approx(0.041)


def test_locates_json_inside_prose() -> None:
    parsed, cost, usage = parse_claude_output(_load("claude_output_prose_wrapped.json"))
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


def test_parses_when_diagnostic_json_precedes_envelope() -> None:
    """Without --bare, claude may emit hook/plugin diagnostic JSON on stdout
    before the result envelope. The parser must locate the envelope (the
    object with a 'result' field) and ignore the diagnostic prefix.
    """
    parsed, cost, usage = parse_claude_output(_load("claude_output_with_diagnostic.json"))
    assert parsed["strategy_id"] == "gen_4"
    assert parsed["one_line_summary"] == "diagnostic prefix test"
    assert cost == pytest.approx(0.018)


def test_parses_when_extra_text_follows_envelope() -> None:
    """Trailing content after a valid envelope must not break parsing."""
    valid_envelope = json.dumps({
        "result": '{"strategy_id":"gen_5","one_line_summary":"tail test","hypothesis":"x","novelty_justification":"x","failure_mode":"x","allow_short":false,"strategy_file":"# x\\n","config_file":"run_name: gen_5\\n"}',
        "total_cost_usd": 0.022,
    })
    stdout = valid_envelope + "\nINFO: shutting down session\n"
    parsed, cost, usage = parse_claude_output(stdout)
    assert parsed["strategy_id"] == "gen_5"
    assert cost == pytest.approx(0.022)


def test_extracts_usage_when_present() -> None:
    """An envelope carrying a `usage` block yields the normalized token dict."""
    parsed, cost, usage = parse_claude_output(_load("claude_output_with_usage.json"))
    assert parsed["strategy_id"] == "gen_usage"
    assert cost == pytest.approx(0.05)
    assert usage == {
        "input": 3120,
        "output": 3540,
        "cache_creation": 0,
        "cache_read": 18000,
    }


def test_usage_is_none_when_envelope_has_no_usage_block() -> None:
    """No `usage` key at all -> usage is None, no error."""
    envelope = json.dumps({
        "result": '{"strategy_id":"gen_x","one_line_summary":"s","hypothesis":"h","novelty_justification":"n","failure_mode":"f","allow_short":false,"strategy_file":"# x\\n","config_file":"run_name: gen_x\\n"}',
        "total_cost_usd": 0.01,
    })
    parsed, cost, usage = parse_claude_output(envelope)
    assert usage is None


def test_usage_missing_subfields_default_to_zero() -> None:
    """`usage` present but a sub-field absent -> that component is 0."""
    envelope = json.dumps({
        "result": '{"strategy_id":"gen_y","one_line_summary":"s","hypothesis":"h","novelty_justification":"n","failure_mode":"f","allow_short":false,"strategy_file":"# x\\n","config_file":"run_name: gen_y\\n"}',
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 500, "output_tokens": 700},
    })
    parsed, cost, usage = parse_claude_output(envelope)
    assert usage == {
        "input": 500,
        "output": 700,
        "cache_creation": 0,
        "cache_read": 0,
    }
