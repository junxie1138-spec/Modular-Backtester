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
    # The old last-30 dedup prompt block has been removed; uniqueness comes
    # from the precomputed slot pool.
    assert "ALREADY-GENERATED IDEAS" not in text
    # Hard contract markers from Appendix A.
    assert "GeneratedStrategy" in text
    assert "shift(1)" in text
    assert "strict JSON" in text


def test_prompt_has_no_dedup_tail_block() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "exit_rule", "constraint_twist", "inspiration_anchor",
    )}
    text = build_prompt(strategy_id="gen_1", slots=slots)
    assert "last 30" not in text.lower()
    assert "(none yet)" not in text


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
