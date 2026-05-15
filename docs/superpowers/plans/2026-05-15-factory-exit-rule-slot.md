# Factory `exit_rule` Inspiration-Matrix Slot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a seventh slot, `exit_rule`, to the factory's inspiration matrix so every generated strategy is assigned an explicit exit mechanic (trailing stop, fixed-bar, signal-reversal, etc.).

**Architecture:** `exit_rule` is a new entry in `SLOT_NAMES` + `SLOTS` in `factory/slots.py`. The two exit-related entries currently in `constraint_twist` migrate into it. A guard in `pull_slots` re-picks `constraint_twist` whenever the drawn pair is incompatible with the drawn `exit_rule`. The prompt template gains a `{{exit_rule}}` line; `build_prompt` substitutes it generically with no code change.

**Tech Stack:** Python 3, `pytest`, `random.Random`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-15-factory-exit-rule-slot-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `factory/slots.py` | The inspiration matrix + `pull_slots` draw logic + guard | Modify |
| `factory/prompt.py` | The Appendix-A prompt template | Modify |
| `factory/tests/test_slots.py` | Slot-matrix and guard tests | Modify |
| `factory/tests/test_prompt.py` | Prompt-substitution tests | Modify |

No other files change. `factory/cycle.py` consumes `pull_slots`/`build_prompt` generically and passes the `slots` dict straight through — it picks up the new slot with no edit. `test_results.py` and `test_dashboard.py` use partial, slot-agnostic dicts as test data and need no change.

---

## Task 1: Add the `exit_rule` slot and migrate exit entries out of `constraint_twist`

**Files:**
- Modify: `factory/slots.py`
- Test: `factory/tests/test_slots.py`

- [ ] **Step 1: Update `test_slots.py` — rename the slot-names test and add three slot tests**

In `factory/tests/test_slots.py`, **rename** `test_six_slots_with_expected_names` to `test_seven_slots_with_expected_names` and replace its body so the expected tuple includes `"exit_rule"`:

```python
def test_seven_slots_with_expected_names() -> None:
    assert SLOT_NAMES == (
        "strategy_family",
        "signal_primitive",
        "holding_horizon",
        "direction",
        "exit_rule",
        "constraint_twist",
        "inspiration_anchor",
    )
    for name in SLOT_NAMES:
        assert len(SLOTS[name]) >= 3, name
```

Then add these three new tests (place them after `test_pull_returns_one_per_slot`):

```python
def test_exit_rule_slot_has_six_values() -> None:
    assert len(SLOTS["exit_rule"]) == 6


def test_exit_entries_migrated_out_of_constraint_twist() -> None:
    twists = SLOTS["constraint_twist"]
    assert "fixed-bar exit (no signal-based exit)" not in twists
    assert "no stop-loss allowed" not in twists
    assert len(twists) == 8


def test_pull_includes_exit_rule() -> None:
    rng = random.Random(123)
    pulled = pull_slots(rng)
    assert pulled["exit_rule"] in SLOTS["exit_rule"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest factory/tests/test_slots.py -q`
Expected: FAIL — `test_seven_slots_with_expected_names` fails (SLOT_NAMES is still the 6-tuple), `test_exit_rule_slot_has_six_values` and `test_pull_includes_exit_rule` fail with `KeyError: 'exit_rule'`, `test_exit_entries_migrated_out_of_constraint_twist` fails (length is 10, entries still present).

- [ ] **Step 3: Implement the slot and migration in `factory/slots.py`**

Replace the entire body of `factory/slots.py` (the `SLOT_NAMES` tuple and `SLOTS` dict) with this. Add the six `_EXIT_*` named constants above `SLOT_NAMES`; insert `"exit_rule"` into `SLOT_NAMES` between `"direction"` and `"constraint_twist"`; add the `"exit_rule"` entry to `SLOTS`; remove `"fixed-bar exit (no signal-based exit)"` and `"no stop-loss allowed"` from `constraint_twist`. Leave `pull_slots` exactly as it is — the guard arrives in Task 2.

```python
from __future__ import annotations

import random
from typing import Mapping

# --- exit_rule slot values (named so SLOTS and the Task-2 guard stay in sync) -
_EXIT_TRAILING_HWM = (
    "rolling-high trailing stop (track the highest close since entry; exit "
    "when close falls k*ATR below that in-trade high-water mark; the stop "
    "only ratchets up)"
)
_EXIT_FIXED_BAR = "fixed-bar exit (exit exactly N bars after entry, no signal-based exit)"
_EXIT_SIGNAL_REVERSAL = "signal-reversal exit (exit only when the entry condition flips)"
_EXIT_PROFIT_TARGET_TIME = (
    "profit-target + time-stop (exit at +X% gain or after N bars, whichever "
    "comes first)"
)
_EXIT_VOL_STOP = (
    "fixed volatility-stop (exit when close falls below entry price minus "
    "k*ATR - fixed, not trailing)"
)
_EXIT_BREAKEVEN_TRAIL = (
    "breakeven-then-trail (after price reaches +X%, move the stop to entry "
    "price, then trail by k*ATR; the stop only ever moves up, never down)"
)

SLOT_NAMES: tuple[str, ...] = (
    "strategy_family",
    "signal_primitive",
    "holding_horizon",
    "direction",
    "exit_rule",
    "constraint_twist",
    "inspiration_anchor",
)

SLOTS: Mapping[str, tuple[str, ...]] = {
    "strategy_family": (
        "momentum", "mean-reversion", "breakout", "volatility-targeting",
        "seasonality", "regime-switching", "range-compression",
        "gap-behavior", "drawdown-recovery", "autocorrelation",
        "relative-position", "trend-strength",
    ),
    "signal_primitive": (
        "close-to-close returns", "high-low range dynamics",
        "volume-confirmed moves", "volatility (std/ATR)",
        "gap (open vs prior close)", "rolling rank/percentile",
        "consecutive-streak count", "distance-from-MA (z-score)",
        "rate-of-change acceleration", "drawdown depth",
    ),
    "holding_horizon": (
        "1-2 days", "3-5 days", "1-2 weeks", "3-4 weeks",
    ),
    "direction": (
        "long-only", "long-only", "long/short",
    ),
    "exit_rule": (
        _EXIT_TRAILING_HWM,
        _EXIT_FIXED_BAR,
        _EXIT_SIGNAL_REVERSAL,
        _EXIT_PROFIT_TARGET_TIME,
        _EXIT_VOL_STOP,
        _EXIT_BREAKEVEN_TRAIL,
    ),
    "constraint_twist": (
        "<=2 tunable params", "regime filter on 200-day MA",
        "signal-scaled position sizing", "symmetric entry/exit rule",
        "two-primitive AND (both must agree)",
        "percentile threshold instead of fixed level",
        "warmup <=10 bars",
        "two-bar confirmation before entry",
    ),
    "inspiration_anchor": (
        "hysteresis control", "predator-prey cycles",
        "queue overflow / capacity limits", "signal-to-noise filtering",
        "spring tension / elastic restoring force",
        "epidemic curves (susceptible-infected)",
        "traffic shockwaves", "elastic vs plastic deformation",
        "refractory period after a spike", "tide tables / standing waves",
    ),
}


def pull_slots(rng: random.Random) -> dict[str, str]:
    """Return one randomly-chosen value per slot."""
    return {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
```

> Note: `_EXIT_VOL_STOP` uses an ASCII hyphen (`k*ATR - fixed`) rather than the em-dash shown in the spec, matching the all-ASCII style of this module and the prompt file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest factory/tests/test_slots.py -q`
Expected: PASS — all tests in the file pass, including the three new ones and the renamed `test_seven_slots_with_expected_names`.

- [ ] **Step 5: Commit**

```bash
git add factory/slots.py factory/tests/test_slots.py
git commit -m "feat(factory): add exit_rule slot, migrate exit twists out of constraint_twist"
```

---

## Task 2: Add the guard against contradictory `(constraint_twist, exit_rule)` draws

**Files:**
- Modify: `factory/slots.py`
- Test: `factory/tests/test_slots.py`

- [ ] **Step 1: Add two guard tests to `test_slots.py`**

In `factory/tests/test_slots.py`, update the import line to also import `_INCOMPATIBLE`:

```python
from factory.slots import SLOT_NAMES, SLOTS, _INCOMPATIBLE, pull_slots
```

Then add these two tests at the end of the file:

```python
def test_guard_never_yields_incompatible_pair() -> None:
    rng = random.Random(2026)
    for _ in range(3000):
        slots = pull_slots(rng)
        pair = (slots["constraint_twist"], slots["exit_rule"])
        assert pair not in _INCOMPATIBLE, pair


def test_guard_preserves_exit_rule_support() -> None:
    # The guard re-picks only constraint_twist, so every exit_rule value
    # must still be reachable. This asserts preserved support, not frequency.
    rng = random.Random(99)
    seen = {pull_slots(rng)["exit_rule"] for _ in range(3000)}
    assert seen == set(SLOTS["exit_rule"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest factory/tests/test_slots.py -q`
Expected: FAIL — the whole file errors at collection with `ImportError: cannot import name '_INCOMPATIBLE' from 'factory.slots'`, because the constant does not exist yet.

- [ ] **Step 3: Implement `_INCOMPATIBLE` and the guard in `factory/slots.py`**

In `factory/slots.py`, add the `_INCOMPATIBLE` frozenset immediately after the `SLOTS` dict (before `pull_slots`):

```python
# (constraint_twist, exit_rule) pairs that must never co-occur.
# - "symmetric entry/exit rule" implies the exit is the logical inverse of the
#   entry, which only holds for the signal-reversal exit.
# - "warmup <=10 bars" is too short for the ATR-based exits to be stable.
_TWIST_SYMMETRIC = "symmetric entry/exit rule"
_TWIST_SHORT_WARMUP = "warmup <=10 bars"

_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {
        (_TWIST_SYMMETRIC, e)
        for e in SLOTS["exit_rule"]
        if e != _EXIT_SIGNAL_REVERSAL
    }
    | {
        (_TWIST_SHORT_WARMUP, e)
        for e in (_EXIT_TRAILING_HWM, _EXIT_VOL_STOP, _EXIT_BREAKEVEN_TRAIL)
    }
)
```

Then replace `pull_slots` with the guarded version:

```python
def pull_slots(rng: random.Random) -> dict[str, str]:
    """Return one randomly-chosen value per slot.

    After drawing all slots, re-pick `constraint_twist` from its compatible
    subset if the drawn (constraint_twist, exit_rule) pair is in _INCOMPATIBLE.
    `exit_rule` is never re-drawn, so its draw distribution is unaffected.
    The compatible subset is never empty: for any single exit_rule value at
    most 2 of the 8 constraint_twist values are forbidden.
    """
    slots = {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
    exit_rule = slots["exit_rule"]
    if (slots["constraint_twist"], exit_rule) in _INCOMPATIBLE:
        compatible = [
            t for t in SLOTS["constraint_twist"]
            if (t, exit_rule) not in _INCOMPATIBLE
        ]
        slots["constraint_twist"] = rng.choice(compatible)
    return slots
```

Confirm the two twist literals already present in `SLOTS["constraint_twist"]` (`"symmetric entry/exit rule"` and `"warmup <=10 bars"`) are byte-identical to `_TWIST_SYMMETRIC` and `_TWIST_SHORT_WARMUP`. Optionally replace those two literals in the `SLOTS` dict with the constants so they cannot drift.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest factory/tests/test_slots.py -q`
Expected: PASS — all tests pass, including `test_guard_never_yields_incompatible_pair` and `test_guard_preserves_exit_rule_support`.

- [ ] **Step 5: Commit**

```bash
git add factory/slots.py factory/tests/test_slots.py
git commit -m "feat(factory): guard exit_rule against contradictory constraint_twist draws"
```

---

## Task 3: Add the `{{exit_rule}}` line to the prompt template

**Files:**
- Modify: `factory/prompt.py`
- Test: `factory/tests/test_prompt.py`

- [ ] **Step 1: Update `test_prompt.py` — add `exit_rule` to the three existing tests and add a substitution test**

In `factory/tests/test_prompt.py`:

(a) In `test_build_prompt_fills_all_placeholders`, add an `exit_rule` entry to the `slots` dict so it reads:

```python
    slots = {
        "strategy_family": "momentum",
        "signal_primitive": "close-to-close returns",
        "holding_horizon": "3-5 days",
        "direction": "long-only",
        "exit_rule": "fixed-bar exit (exit exactly N bars after entry, no signal-based exit)",
        "constraint_twist": "<=2 tunable params",
        "inspiration_anchor": "hysteresis control",
    }
```

(b) In both `test_empty_dedup_tail_is_handled` and `test_long_dedup_tail_caps_at_30`, add `"exit_rule"` to the slot-name tuple so each reads:

```python
    slots = {n: "x" for n in (
        "strategy_family", "signal_primitive", "holding_horizon",
        "direction", "exit_rule", "constraint_twist", "inspiration_anchor",
    )}
```

(c) Add this new test at the end of the file:

```python
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
    text = build_prompt(strategy_id="gen_1", slots=slots, dedup_tail=[])
    assert "DISTINCTIVE-EXIT-RULE-MARKER" in text
    assert "{{exit_rule}}" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest factory/tests/test_prompt.py -q`
Expected: FAIL — `test_exit_rule_placeholder_is_substituted` fails (`"DISTINCTIVE-EXIT-RULE-MARKER"` is not in the output because the template has no `{{exit_rule}}` placeholder), and `test_build_prompt_fills_all_placeholders` fails on `assert v in text` for the `exit_rule` value (same reason).

- [ ] **Step 3: Add the `{{exit_rule}}` block to `PROMPT_TEMPLATE` in `factory/prompt.py`**

In `factory/prompt.py`, find this passage in `PROMPT_TEMPLATE` (inside the "THIS IDEA'S RANDOM CONSTRAINTS" block):

```
- Direction: {{direction}} (if "long/short", you may emit -1 signals; if
  "long-only", never emit -1)
- Hard twist (must satisfy): {{constraint_twist}}
```

Replace it with:

```
- Direction: {{direction}} (if "long/short", you may emit -1 signals; if
  "long-only", never emit -1)
- Exit rule (the strategy MUST implement this exit mechanic): {{exit_rule}}
  Implement it inside generate_signals by driving df["signal"] to 0 when the
  exit fires - there is no config-level stop-loss block. A bar-indexed Python
  loop is acceptable for this exit computation: trailing and breakeven exits
  are path-dependent and have no clean vectorised equivalent.
- Hard twist (must satisfy): {{constraint_twist}}
```

`build_prompt` already substitutes every `slots` key via `filled.replace("{{" + name + "}}", value)`, so no function change is needed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest factory/tests/test_prompt.py -q`
Expected: PASS — all tests in the file pass, including the new `test_exit_rule_placeholder_is_substituted`.

- [ ] **Step 5: Commit**

```bash
git add factory/prompt.py factory/tests/test_prompt.py
git commit -m "feat(factory): surface exit_rule slot in the generation prompt"
```

---

## Task 4: Full factory regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the entire factory test suite**

Run: `python -m pytest factory/tests/ -q`
Expected: PASS — every test passes with zero regressions. In particular `factory/tests/test_cycle.py`, `factory/tests/test_results.py`, and `factory/tests/test_dashboard.py` still pass: `cycle.py` consumes `pull_slots`/`build_prompt` generically, and the `test_results.py`/`test_dashboard.py` slot dicts are slot-agnostic test data.

- [ ] **Step 2: If any test fails, stop and investigate**

A failure here means something outside the 4 planned files depends on the slot count or the migrated `constraint_twist` entries. Do not paper over it — diagnose the failing test, confirm whether the spec's 4-file scope was wrong, and report before continuing.

- [ ] **Step 3: No commit**

This task changes no files. If Step 1 passed, the feature is complete.

---

## Self-Review

**Spec coverage:**
- Spec §1.1 (slot position) → Task 1 Step 3 (`SLOT_NAMES` with `exit_rule` between `direction` and `constraint_twist`), asserted by Task 1 Step 1's `test_seven_slots_with_expected_names`.
- Spec §1.2 (6 values) → Task 1 Step 3 (`_EXIT_*` constants + `SLOTS["exit_rule"]`), asserted by `test_exit_rule_slot_has_six_values`.
- Spec §1.3 (E1/E6 wording clarifications) → encoded verbatim in the `_EXIT_TRAILING_HWM` and `_EXIT_BREAKEVEN_TRAIL` constants.
- Spec §2 (migration, `constraint_twist` → 8 values) → Task 1 Step 3, asserted by `test_exit_entries_migrated_out_of_constraint_twist`.
- Spec §3.1/§3.2 (guard, `_INCOMPATIBLE`, re-pick `constraint_twist`) → Task 2, asserted by `test_guard_never_yields_incompatible_pair`.
- Spec §3.2 (named constants decision) → Task 1/Task 2 use `_EXIT_*` and `_TWIST_*` constants.
- Spec §3.3 (soft rule, unenforced) → intentionally not implemented; documented in spec only. No task — correct.
- Spec §4 (prompt `{{exit_rule}}` line) → Task 3.
- Spec §6 (test changes) → Tasks 1–3 Step 1 each; `test_guard_preserves_exit_rule_support` name matches spec §6.1.5.
- Spec §8 acceptance criteria 1–6 → criteria 1–4 covered by Task 1/Task 2 tests, criterion 5 by Task 3, criterion 6 by Tasks 1–4 test runs.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step shows complete code.

**Type consistency:** `_EXIT_TRAILING_HWM`, `_EXIT_FIXED_BAR`, `_EXIT_SIGNAL_REVERSAL`, `_EXIT_PROFIT_TARGET_TIME`, `_EXIT_VOL_STOP`, `_EXIT_BREAKEVEN_TRAIL`, `_TWIST_SYMMETRIC`, `_TWIST_SHORT_WARMUP`, and `_INCOMPATIBLE` are used consistently across Tasks 1, 2, and the test imports. `pull_slots` keeps its `(rng: random.Random) -> dict[str, str]` signature throughout.

One scope note surfaced during review: the spec says "no change to `constraint_twist` literals", but Task 2 Step 3 optionally swaps the two twist literals in `SLOTS["constraint_twist"]` for `_TWIST_*` constants. This is a DRY safety measure (it guarantees the guard's twist strings match the slot's) and is presented as optional — it does not alter any drawn value.
