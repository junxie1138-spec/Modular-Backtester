# Factory: swap decision metric from Sharpe to Sortino

- **Date:** 2026-05-17
- **Status:** Design — awaiting review
- **Scope:** `factory/` strategy factory only. The core backtester already
  computes Sortino; no `backtester/` changes.

## Context

The strategy factory currently ranks, alerts on, and promotes generated
strategies using the out-of-sample Sharpe ratio (`wfo.oos_sharpe`). Sharpe
penalises upside and downside volatility equally. Sortino penalises only
downside deviation, which better matches "is this strategy good" without
punishing strategies for winning unevenly.

The backtester already computes Sortino: `backtester/analytics/metrics.py`
defines `sortino_ratio`, `compute_summary_metrics` emits a `sortino` key, and
`"sortino"` is a registered optimisation objective in
`backtester/optimize/objectives.py`. So this is a factory-side rewiring, not
new metric maths.

A separate idea — switching the factory to an hourly timeframe — was
considered alongside this and **deliberately deferred** (see Non-goals). It
is a milestone-sized effort: every annualised metric is hardcoded to daily
via `TRADING_DAYS_PER_YEAR`, and hourly data sourcing has a ~730-day yfinance
history limit.

## Goal

After this change, the factory selects, alerts on, and promotes strategies
using out-of-sample Sortino instead of Sharpe, end to end:

- In-sample WFO optimisation objective becomes `sortino`.
- The factory record carries `wfo.oos_sortino`.
- The alert gate and promotion gate evaluate Sortino.
- `oos_sharpe` is retained in the record but demoted to a display-only field.

## Decisions (locked with the user)

1. **Thresholds unchanged.** `alert_threshold` stays `1.0`; the promotion
   threshold stays `0.7`. Sortino typically reads higher than Sharpe, so the
   gate is temporarily looser; the user will retune from observed Sortino
   distributions later. No threshold tuning in this change.
2. **In-sample objective flips to Sortino.** The generation prompt template
   instructs `objective: sortino`, so in-sample param selection and the OOS
   decision metric agree.
3. **`oos_sharpe` is retained, not removed.** See the explicit statement
   below.

## Non-goals

- **Hourly timeframe.** Deferred to its own milestone. Captured as a backlog
  idea, not built here.
- **Drawdown veto gate.** Considered, declined for this change ("Sortino swap
  only" scope). May revisit.
- **Dashboard OOS metric rework.** The dashboard keeps its existing
  `OOS Sharpe` column. Adding an `OOS Sortino` column / re-labelling the
  decision metric on the dashboard is deferred. (The *mandatory* dashboard
  touches below are a separate, smaller matter — see Change 7.)
- **`backtester/wfo/multi_stitcher.py`.** Its hand-built `oos_summary` omits
  `sortino`. The factory only runs single-symbol WFO (routed through
  `stitcher.py`, which emits the full `compute_summary_metrics` dict), so this
  is not exercised today. If the factory ever runs multi-symbol WFO,
  `multi_stitcher.py` must add `sortino` first.

## `oos_sharpe` after the swap — explicit status

> **`oos_sharpe`: retained in the WFO record as a stored field. After this
> change it is NOT used in any gate or alert evaluation. Its only consumers
> are display/logging: the cycle completion log line, the Telegram alert
> message body, and the dashboard. Do not re-wire it into alert, screening,
> or promotion logic — the decision metric is `oos_sortino`.**

Consumer audit (verified against the current tree):

| Consumer | Path | Role after swap |
|---|---|---|
| `parse_wfo_summary` writes it | `factory/stages.py` | Source — retained |
| Cycle completion log | `factory/cycle.py` | Display/log only |
| Telegram alert body | `factory/notify.py` | Display only |
| Dashboard table + sort | `factory/dashboard/**` | Display only |
| Alert gate | `factory/notify.py` `maybe_send_alert` | Reads `alert_threshold_metric` from config — **not** hardcoded to `oos_sharpe`; switches via `settings.toml` |
| Promotion trigger | `factory/cycle.py` | Reads `s.promotion.trigger_metric` from config — **not** hardcoded; switches via `settings.toml` |
| Promotion per-ticker aggregation | `factory/promote.py:167` | **Currently hardcoded to `oos_sharpe` — this IS a gate and is moved to `oos_sortino`** |

## Changes

### 1. `factory/stages.py` — emit `oos_sortino`
`parse_wfo_summary` adds `"oos_sortino": float(oos["sortino"])` to the parsed
record, alongside the existing `oos_sharpe` (retained). The live WFO
`summary.json` already carries `sortino` in `oos_summary` (via
`compute_summary_metrics` in `stitcher.py`).

### 2. `factory/config/settings.toml`
- `alert_threshold_metric` → `"wfo.oos_sortino"` (`alert_threshold` stays `1.0`).
- `[promotion] trigger_metric` → `"wfo.oos_sortino"`.
- `[promotion]` key `min_avg_sharpe` → renamed `min_avg_sortino`, value stays `0.7`.

### 3. `factory/settings_loader.py`
- `PromotionCfg.min_avg_sharpe` field → `min_avg_sortino`.
- Loader line and default updated; `trigger_metric` default → `"wfo.oos_sortino"`.

### 4. `factory/promote.py`
- Per-ticker aggregation reads `oos_sortino` instead of `oos_sharpe`
  (currently `factory/promote.py:167`).
- Gate compares against `promotion_cfg.min_avg_sortino`.
- `PromotionResult.avg_sharpe` → `avg_sortino`;
  `min_avg_sharpe_threshold` → `min_avg_sortino_threshold`.
- Module docstring and per-ticker log line updated to say Sortino.

### 5. `factory/prompt.py` — in-sample objective
Config template (`factory/prompt.py:128`): `objective: sharpe` →
`objective: sortino`.

### 6. `factory/cycle.py`
- Update the stale comment ("alert trigger is unchanged (still
  `wfo.oos_sharpe` by default)") to reference Sortino.
- Cycle completion log line logs `oos_sortino` (the decision metric) in
  place of `oos_sharpe`.
- Promotion log line: `avg_sharpe` → `avg_sortino` field name.

### 7. `factory/notify.py` + dashboard read-sites (mandatory)
- `format_alert_message`: the alert now fires on Sortino, so the message
  headlines **OOS Sortino**. The existing `OOS Sharpe` line is kept below it
  as secondary info.
- Dashboard sites that read the **renamed** `PromotionResult` fields must be
  updated or the page breaks: `record.promotion.avg_sharpe` /
  `min_avg_sharpe_threshold` in `detail.html`, `overview.html`, and
  `overview.js`. This is mandatory correctness work, distinct from the
  deferred OOS-column rework. The per-ticker `p.oos_sharpe` table in
  `detail.html` keeps working because `oos_sharpe` is retained.

### 8. Tests and fixtures
- `factory/tests/fixtures/sample_wfo_summary.json`: add a `sortino` key to
  `oos_summary`.
- Update `test_stages_parsers.py` (new `oos_sortino` key),
  `test_promote.py` (renamed result fields, Sortino aggregation),
  `test_settings_loader.py` (renamed `min_avg_sortino`),
  `test_notify.py` (alert body), `test_cycle.py` as needed.

## Testing

- Unit: `parse_wfo_summary` emits `oos_sortino`; `promote.py` aggregates
  Sortino and gates on `min_avg_sortino`; settings loader parses the renamed
  key; alert message renders Sortino.
- Run the full `factory/tests/` suite — it must stay green.
- Manual: one factory cycle end to end, confirm the record contains
  `wfo.oos_sortino`, the cycle log shows Sortino, and the dashboard renders
  without template errors.

## Risks

- **Looser gate.** Keeping thresholds at Sharpe-era values means more
  strategies clear the alert/promotion gate until retuned. Accepted, by
  decision 1.
- **Renamed config key.** Any existing `settings.local.toml` overriding
  `[promotion] min_avg_sharpe` would be silently ignored after the rename.
  Check the live `settings.local.toml` during implementation; the current one
  does not override it.
- **Dashboard cosmetic gap.** The dashboard still labels the column
  `OOS Sharpe` though the decision metric is now Sortino. Known, deferred per
  Non-goals.
