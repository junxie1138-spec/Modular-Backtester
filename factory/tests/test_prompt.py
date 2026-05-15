from factory.prompt import build_prompt


def test_build_prompt_fills_all_placeholders() -> None:
    slots = {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }
    text = build_prompt(
        strategy_id="gen_1715800000",
        slots=slots,
        dedup_tail=["sma cross 50/200", "rsi mean reversion 14"],
    )
    # Every placeholder should be filled.
    assert "{{" not in text and "}}" not in text
    # Slot values present in the prompt.
    for v in slots.values():
        assert v in text
    # Strategy id appears in the prompt (must match injected value).
    assert "gen_1715800000" in text
    # Dedup tail appears as numbered/bulleted lines.
    assert "sma cross 50/200" in text
    assert "rsi mean reversion 14" in text
    # Hard contract markers from Appendix A.
    assert "GeneratedStrategy" in text
    assert "shift(1)" in text
    assert "strict JSON" in text


def test_empty_dedup_tail_is_handled() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "constraint_twist", "inspiration_anchor",
    )}
    text = build_prompt(strategy_id="gen_1", slots=slots, dedup_tail=[])
    assert "(none yet)" in text


def test_long_dedup_tail_caps_at_30() -> None:
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "constraint_twist", "inspiration_anchor",
    )}
    tail = [f"idea {i}" for i in range(100)]
    text = build_prompt(strategy_id="gen_1", slots=slots, dedup_tail=tail)
    # Only the last 30 of 100 should appear in the prompt.
    assert "idea 70" in text
    assert "idea 99" in text
    assert "idea 69" not in text
