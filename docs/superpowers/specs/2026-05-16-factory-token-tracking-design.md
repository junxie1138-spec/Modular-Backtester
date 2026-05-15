# Strategy Factory — token tracking — design spec

**Status:** Approved (design phase). Implementation plan to be authored next via the `superpowers:writing-plans` skill.

**Goal:** Record the token consumption of every strategy-generation call and surface it on the factory dashboard, alongside the dollar cost that is already tracked.

**Motivation:** The factory records `generation_cost_usd` per strategy but discards token counts entirely. A user asking "how many tokens does each strategy consume?" currently has no answer. Tokens are already present in the `claude -p` output — they are simply thrown away.

**Approach in one sentence:** Token counts follow the exact path `generation_cost_usd` already takes — captured in `factory/generate.py`, persisted on the results record, surfaced on the dashboard — adding no new subsystem.

---

## 1. Background — what is tracked today

`factory/generate.py:parse_claude_output` parses the `claude -p --output-format json` envelope. It extracts `total_cost_usd` and **nothing else** from the envelope's accounting fields. The `usage` block (token counts) is discarded.

The cost then flows: `GenerationResult.cost_usd` → `factory/cycle.py:run_cycle` → `factory/results.py:build_record` / `build_failed_record` as the `generation_cost_usd` field → results shard → dashboard, which shows it per-strategy on the detail view and as a cumulative total on the overview.

This spec adds a parallel `generation_tokens` field that travels the same path. Nothing about the cost field changes.

---

## 2. Capture — `factory/generate.py`

`parse_claude_output` additionally extracts the envelope's `usage` block and returns it next to the cost.

- The CLI envelope's `usage` object reports four token counts. **The exact field names must be verified against a live `claude -p --output-format json` invocation during implementation** — they are expected to be `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, but the implementation must confirm this rather than assume it.
- Extraction is **defensive**: if the envelope has no `usage` block at all, the captured value is `None`. If `usage` is present but a given sub-field is absent, that component is treated as `0`.
- `GenerationResult` gains a field `usage: dict | None` carrying the normalized token dict (see §3 for the shape) or `None`.

This mirrors `cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)` — the same envelope, one more `.get`.

---

## 3. Record schema — `factory/results.py`

`build_record` and `build_failed_record` gain a `generation_tokens` field. Its value is either `None` or a nested object with **all four raw counts retained**:

```json
"generation_tokens": {
  "input": 3120,
  "output": 3540,
  "cache_creation": 0,
  "cache_read": 18000
}
```

- The nested-object shape matches the existing record convention — `backtest`, `optimize`, `wfo`, and `promotion` are already nested sub-objects on the record.
- **All four counts are stored raw and kept separate.** The UI folds `cache_creation` and `cache_read` into a single displayed "cached" figure (§5), but the record itself always retains both values independently — so a future consumer can distinguish cache-creation from cache-read without a data migration.
- `generation_tokens` is `None` when generation did not yield a usable `usage` block. This is the same rule `generation_cost_usd` already follows: a generation that times out or whose output cannot be parsed records `0.0` cost today and will record `None` tokens. Capturing tokens on a partial/failed generation is explicitly **out of scope** (see §7) — tokens track cost's behavior exactly, no better and no worse.
- **Backwards compatibility:** results records written before this change have no `generation_tokens` key. All readers treat a missing key as `None`, identical to an explicit `None`.

---

## 4. Cycle wiring — `factory/cycle.py`

`run_cycle` threads `gen.usage` into `build_record` and `build_failed_record` at exactly the call sites where it already passes `gen.cost_usd` / `generation_cost_usd`. No change to cycle control flow, stage order, or any other behavior.

---

## 5. Dashboard — `factory/dashboard/`

Two surfaces, mirroring exactly where `generation_cost_usd` already appears.

### 5.1 Detail view (`detail.html`)

Directly beneath the existing "Generation cost" line, display three token lines:

- **input:** `<generation_tokens.input>`
- **output:** `<generation_tokens.output>`
- **cached:** `<generation_tokens.cache_creation + generation_tokens.cache_read>`

The "cached" figure is the **sum of `cache_creation` and `cache_read`**. The two raw fields are not shown as separate UI elements now; a later iteration may add a parenthetical or tooltip noting that "cached" = `cache_creation + cache_read`. The raw record retains both values regardless (§3).

When `generation_tokens` is `None` (failed generation, or a pre-change record), the detail view shows "Generation tokens: n/a" instead of the three lines.

### 5.2 Overview (`overview.html` + `server.py:_aggregate`)

Add a **"Cumulative tokens"** metric beside the existing "Cumulative spend".

The cumulative-tokens total is defined explicitly as:

> **total tokens = input + output + cache_creation + cache_read**, summed over every results record.

Cached tokens (both `cache_creation` and `cache_read`) **do** count toward the cumulative total — there is no ambiguity: every token the model billed is included. Records with `generation_tokens = None` contribute `0`.

Because `factory/results.py:read_records` already unions every machine's shards, this total is automatically pool-wide across a distributed factory — no extra work.

No per-row token column is added to the overview table; it stays as it is.

---

## 6. File layout

| File | Action |
|---|---|
| `factory/generate.py` | modify — `parse_claude_output` returns `usage`; `GenerationResult` gains `usage` field |
| `factory/results.py` | modify — `build_record` / `build_failed_record` gain `generation_tokens` |
| `factory/cycle.py` | modify — thread `gen.usage` into the record builders |
| `factory/dashboard/server.py` | modify — `_aggregate` sums cumulative tokens |
| `factory/dashboard/templates/detail.html` | modify — input/output/cached lines |
| `factory/dashboard/templates/overview.html` | modify — "Cumulative tokens" metric |
| `factory/tests/test_generate.py` | modify/extend — envelope with and without `usage` |
| `factory/tests/test_results.py` | modify/extend — `generation_tokens` present and `None` in both builders |
| `factory/tests/test_dashboard.py` | modify/extend — detail breakdown, cumulative sum, missing-field → n/a / 0 |

---

## 7. Out of scope

- **Token capture on failed generations.** Tokens follow `generation_cost_usd` exactly: `None` whenever generation did not return a usable envelope. Recovering partial token usage from a generation that failed at the inner-JSON layer is not addressed (cost is not recovered there today either).
- **Per-row token column on the overview table.** Cumulative total only on the overview; full breakdown only on the detail view.
- **Exposing `cache_creation` and `cache_read` as separate UI elements.** The UI folds them into "cached"; the raw record keeps them separate for the future.
- **Token-based alerting, budgets, or quotas.** This spec records and displays; it does not gate on tokens.

---

## 8. Acceptance criteria

1. `parse_claude_output` returns the `usage` token counts; `GenerationResult` carries them; a missing `usage` block yields `None` without error.
2. `build_record` and `build_failed_record` write a `generation_tokens` object — `{input, output, cache_creation, cache_read}`, all four retained — or `None`.
3. Results records written before this change (no `generation_tokens` key) are read as `None` everywhere, with no error.
4. The dashboard detail view shows input / output / cached token lines, where **cached = cache_creation + cache_read**, and "n/a" when `generation_tokens` is `None`.
5. The dashboard overview shows a "Cumulative tokens" metric defined as **input + output + cache_creation + cache_read summed over all records**, pool-wide across shards, with `None` records contributing 0.
6. The full factory test suite passes, including the extended `test_generate.py`, `test_results.py`, and `test_dashboard.py` coverage.
