from factory.prompt import build_prompt


def test_build_prompt_fills_all_placeholders() -> None:
    slots = {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "exit_rule": "fixed-bar exit (exit exactly N bars after entry, no signal-based exit)",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }
    text = build_prompt(
        strategy_id="gen_1715800000",
        slots=slots,
    )
    # Every placeholder should be filled.
    assert "{{" not in text and "}}" not in text
    # Slot values present in the prompt.
    for v in slots.values():
        assert v in text
    # Strategy id appears in the prompt (must match injected value).
    assert "gen_1715800000" in text
    assert "ALREADY-GENERATED IDEAS" in text
    # Hard contract markers from Appendix A.
    assert "GeneratedStrategy" in text
    assert "shift(1)" in text
    assert "strict JSON" in text


def test_prompt_shows_recent_summaries() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "exit_rule", "constraint_twist", "inspiration_anchor",
    )}
    text = build_prompt(strategy_id="gen_1", slots=slots, recent_summaries=["older idea", "new idea"])
    assert "last 30" in text.lower()
    assert "- older idea\n- new idea" in text


def test_exit_rule_placeholder_is_substituted() -> None:
    slots = {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "exit_rule": "DISTINCTIVE-EXIT-RULE-MARKER",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }
    text = build_prompt(strategy_id="gen_1", slots=slots)
    assert "DISTINCTIVE-EXIT-RULE-MARKER" in text
    assert "{{exit_rule}}" not in text
