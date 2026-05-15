# Factory `exit_rule` inspiration-matrix slot — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Add a seventh slot, `exit_rule`, to the factory's inspiration matrix (the `SLOTS` dict in `factory/slots.py`). Each generation run draws one exit mechanic, which the LLM must implement inside the generated strategy's `generate_signals`. The two exit-related entries currently living in `constraint_twist` migrate into the new slot, and a guard prevents the matrix from drawing a contradictory `(exit_rule, constraint_twist)` pair.

**Scope:** 4 files — `factory/slots.py`, `factory/prompt.py`, `factory/tests/test_slots.py`, `factory/tests/test_prompt.py`. No change to the backtester, the generated config shape, or `build_prompt`'s substitution logic.

**Context:** The factory generates signal-only strategies (`df["signal"]` in `{-1,0,1}`, shifted one bar). The generated config template in `prompt.py` is fixed and has **no** stop-loss / trailing-stop block — so an exit rule cannot be expressed via config. It must be implemented in the strategy's signal logic. This slot makes the exit mechanic an explicit, varied axis of the matrix rather than an occasional `constraint_twist` draw.

---

## 1. The `exit_rule` slot

### 1.1 Slot position

`"exit_rule"` is inserted into `SLOT_NAMES` after `"direction"` and before `"constraint_twist"`:

```python
SLOT_NAMES = (
    "strategy_family",
    "signal_primitive",
    "holding_horizon",
    "direction",
    "exit_rule",          # NEW
    "constraint_twist",
    "inspiration_anchor",
)
```

The matrix grows from 6 slots to 7. `pull_slots` already iterates `SLOT_NAMES` generically, so it picks up the new slot automatically.

### 1.2 Slot values

`SLOTS["exit_rule"]` is a 6-tuple:

| ID | Value string |
|----|--------------|
| E1 | `rolling-high trailing stop (track the highest close since entry; exit when close falls k*ATR below that in-trade high-water mark; the stop only ratchets up)` |
| E2 | `fixed-bar exit (exit exactly N bars after entry, no signal-based exit)` |
| E3 | `signal-reversal exit (exit only when the entry condition flips)` |
| E4 | `profit-target + time-stop (exit at +X% gain or after N bars, whichever comes first)` |
| E5 | `fixed volatility-stop (exit when close falls below entry price minus k*ATR — fixed, not trailing)` |
| E6 | `breakeven-then-trail (after price reaches +X%, move the stop to entry price, then trail by k*ATR; the stop only ever moves up, never down)` |

E2 and E3 are the **migrated** entries (see §2).

### 1.3 Wording clarifications

Two values carry semantics that are easy to implement subtly wrong; their wording is deliberately explicit.

- **E1 — "highest close", not "highest high".** The high-water mark is tracked on the **close** series, not the bar `high`. Rationale: the generated contract is close-based and one-bar-shifted (the strategy decides on bar N's close, fills on N+1). Tracking the HWM on `high` would mix an intrabar quantity into a close-driven, lookahead-safe pipeline. Closes only — no intrabar logic.
- **E6 — activation is gated and the stop is monotonic.** The trail is **dormant** until the peak gain since entry first reaches `+X%` (peak gain = `highest_close_since_entry / entry_price - 1`). At the bar that trigger first fires, the stop jumps to the entry price (breakeven). From then on the stop trails at `max(entry_price, highest_close_since_entry - k*ATR)` and is **monotonically non-decreasing** — it never moves back down. Before the trigger, no stop is active.

---

## 2. Migration out of `constraint_twist`

Two entries are **removed** from `SLOTS["constraint_twist"]`; their intent now lives in the `exit_rule` slot:

| Removed from `constraint_twist` | Now expressed as |
|---|---|
| `fixed-bar exit (no signal-based exit)` | E2 |
| `no stop-loss allowed` | E3 (`signal-reversal exit`) |

`constraint_twist` drops from 10 values to 8. The remaining 8:

```
<=2 tunable params
regime filter on 200-day MA
signal-scaled position sizing
symmetric entry/exit rule
two-primitive AND (both must agree)
percentile threshold instead of fixed level
warmup <=10 bars
two-bar confirmation before entry
```

After migration, `constraint_twist` is purely about structural constraints; `exit_rule` owns all exit mechanics. There is no longer any way to draw two slots that both dictate an exit.

> Note: the migrated E3 is named `signal-reversal exit (exit only when the entry condition flips)` — the original "no stop-loss" phrasing is dropped. Once `exit_rule` is the single source of truth for exits, "no stop-loss" is redundant, and the negative phrasing risks implying that *other* exit rules permit layered stops outside the slot.

---

## 3. Guard against contradictory draws

After migration, `exit_rule` and `constraint_twist` can still collide on two remaining `constraint_twist` values. `pull_slots` enforces a guard.

### 3.1 Enforced incompatibilities

| `constraint_twist` value | Incompatible `exit_rule` values | Reason |
|---|---|---|
| `symmetric entry/exit rule` | E1, E2, E4, E5, E6 (everything except E3) | A symmetric rule means the exit is the logical inverse of the entry. Only `signal-reversal exit` (E3) is genuinely symmetric; time stops, ATR stops, and profit targets are all asymmetric. |
| `warmup <=10 bars` | E1, E5, E6 (the ATR-based exits) | ATR needs lookback stability; an ATR-based stop under a ≤10-bar warmup is brittle or nonsensical. |

### 3.2 Mechanism

`pull_slots` draws all 7 slots, then — if the drawn `(constraint_twist, exit_rule)` pair is incompatible — **re-picks `constraint_twist`** from its compatible subset. `exit_rule` is held fixed (it is the feature axis we want represented); only `constraint_twist` is re-drawn.

Re-selection is a **direct choice from the filtered list**, not a retry loop — so it cannot loop and cannot fail. For any single `exit_rule` value at most 2 of the 8 `constraint_twist` values are forbidden, leaving ≥6 compatible — the compatible subset is never empty.

Reference shape:

```python
# (constraint_twist, exit_rule) pairs that must never co-occur.
_INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset({
    ("symmetric entry/exit rule", E1), ("symmetric entry/exit rule", E2),
    ("symmetric entry/exit rule", E4), ("symmetric entry/exit rule", E5),
    ("symmetric entry/exit rule", E6),
    ("warmup <=10 bars", E1), ("warmup <=10 bars", E5), ("warmup <=10 bars", E6),
})

def pull_slots(rng):
    slots = {name: rng.choice(SLOTS[name]) for name in SLOT_NAMES}
    exit_rule = slots["exit_rule"]
    if (slots["constraint_twist"], exit_rule) in _INCOMPATIBLE:
        compatible = [t for t in SLOTS["constraint_twist"]
                      if (t, exit_rule) not in _INCOMPATIBLE]
        slots["constraint_twist"] = rng.choice(compatible)
    return slots
```

**Implementation decision:** `slots.py` defines the 6 `exit_rule` strings as named module-level constants (e.g. `_EXIT_TRAILING_HWM`, `_EXIT_FIXED_BAR`, … corresponding to E1..E6). Both `SLOTS["exit_rule"]` and `_INCOMPATIBLE` reference those constants by name rather than repeating the literal strings, so the two stay in sync and `_INCOMPATIBLE` is readable. The `E1`..`E6` labels used throughout this spec map one-to-one onto those constants.

### 3.3 Documented soft rule (NOT enforced)

`holding_horizon = "1-2 days"` pairs poorly with the trailing exits **E1** and **E6**: a 1–2 day hold is too short for a trail to meaningfully activate or advance. This is recorded here as a known weak combination. It is **not** enforced by the guard — the matrix may still draw it, and the LLM is expected to cope. If empirically this produces low-quality strategies, promoting it to an enforced rule is a contained follow-up.

---

## 4. Prompt change (`factory/prompt.py`)

A `{{exit_rule}}` line is added to the "THIS IDEA'S RANDOM CONSTRAINTS" block, after the `Direction` line and before the `Hard twist` line:

```
- Direction: {{direction}} (if "long/short", you may emit -1 signals; if
  "long-only", never emit -1)
- Exit rule (the strategy MUST implement this exit mechanic): {{exit_rule}}
  Implement it inside generate_signals by driving df["signal"] to 0 when the
  exit fires — there is no config-level stop-loss block. A bar-indexed Python
  loop is acceptable for this exit computation: trailing and breakeven exits
  are path-dependent and have no clean vectorised equivalent.
- Hard twist (must satisfy): {{constraint_twist}}
```

No change to `build_prompt` — it already substitutes every `slots` key generically via `filled.replace("{{" + name + "}}", value)`. Adding `exit_rule` to the `slots` dict plus the `{{exit_rule}}` placeholder to the template fully wires it.

---

## 5. File layout

| File | Action |
|---|---|
| `factory/slots.py` | modify — add `exit_rule` to `SLOT_NAMES` + `SLOTS`; remove 2 entries from `constraint_twist`; add `_INCOMPATIBLE` + guard in `pull_slots` |
| `factory/prompt.py` | modify — add the `{{exit_rule}}` block to `PROMPT_TEMPLATE` |
| `factory/tests/test_slots.py` | modify — update hard-coded slot tuple; add slot/migration/guard tests |
| `factory/tests/test_prompt.py` | modify — add `exit_rule` to the two hard-coded slot dicts; assert `{{exit_rule}}` substitution |

No changes to: `build_prompt` logic, `factory/generate.py`, the generated config template/shape, the backtester, or any existing generated strategy.

---

## 6. Tests

### 6.1 `factory/tests/test_slots.py`

**Modify existing:**
- `test_six_slots_with_expected_names` → rename to `test_seven_slots_with_expected_names`; update the hard-coded tuple to include `"exit_rule"` in its §1.1 position. The `len(SLOTS[name]) >= 3` loop continues to cover the new slot (6 values).

**Add:**
1. `test_exit_rule_slot_has_six_values` — `len(SLOTS["exit_rule"]) == 6`.
2. `test_exit_entries_migrated_out_of_constraint_twist` — `"fixed-bar exit (no signal-based exit)"` and `"no stop-loss allowed"` are absent from `SLOTS["constraint_twist"]`; `len(SLOTS["constraint_twist"]) == 8`.
3. `test_pull_includes_exit_rule` — `pull_slots` output has an `exit_rule` key with a value in `SLOTS["exit_rule"]`.
4. `test_guard_never_yields_incompatible_pair` — over many seeded draws (e.g. 3000), assert no `(constraint_twist, exit_rule)` pair is in `_INCOMPATIBLE`.
5. `test_guard_preserves_exit_rule_support` — over many seeded draws, all 6 `exit_rule` values appear. This proves the guard preserves *support* (every value stays reachable); it does not assert frequency stability, which is unnecessary since the guard re-picks only `constraint_twist` and so does not directly alter `exit_rule` draws.

### 6.2 `factory/tests/test_prompt.py`

**Modify existing:**
- `test_build_prompt_fills_all_placeholders` — add `"exit_rule": <some value>` to the `slots` dict. The existing `assert "{{" not in text` and the `for v in slots.values()` loop then cover the new placeholder.
- `test_empty_dedup_tail_is_handled` and `test_long_dedup_tail_caps_at_30` — add `"exit_rule"` to their `slots` name lists so `build_prompt` receives all 7 keys.

**Add:**
1. `test_exit_rule_placeholder_is_substituted` — build a prompt with a distinctive `exit_rule` value; assert that value appears in the output and `{{exit_rule}}` does not.

---

## 7. Out of scope

- **Config-level exit rules.** Wiring the generated config template to the backtester's existing `trailing_stop.py` execution-layer stop. The factory contract is signal-only by design; this stays so.
- **Enforcing the soft rule** (`holding_horizon = "1-2 days"` × trailing exits). Documented in §3.3, deliberately unenforced.
- **Validating that the generated strategy actually implements the drawn `exit_rule`.** The factory's existing static/functional validators are unchanged; the exit rule is a prompt-level instruction, not a validated contract.
- **Adding exit-rule parameters to WFO / optimization search.** The exit's `N`, `k`, `X%` are ordinary strategy params chosen by the LLM; no new optimizer surface.

---

## 8. Acceptance criteria

1. `factory/slots.py` exposes a 7-entry `SLOT_NAMES` with `"exit_rule"` between `"direction"` and `"constraint_twist"`; `SLOTS["exit_rule"]` has the 6 values from §1.2.
2. `SLOTS["constraint_twist"]` no longer contains `"fixed-bar exit (no signal-based exit)"` or `"no stop-loss allowed"` and has exactly 8 values.
3. `pull_slots` never returns a `(constraint_twist, exit_rule)` pair listed in `_INCOMPATIBLE` — verified across ≥3000 seeded draws.
4. All 6 `exit_rule` values remain reachable from `pull_slots`. The guard re-picks only `constraint_twist`, so it does not directly alter `exit_rule` draws — the spec asserts preserved *support*, not frequency stability.
5. `build_prompt` produces a prompt containing the drawn `exit_rule` value, with no leftover `{{exit_rule}}` placeholder.
6. `python -m pytest factory/tests/test_slots.py factory/tests/test_prompt.py -q` passes with zero regressions, including the new tests in §6.
