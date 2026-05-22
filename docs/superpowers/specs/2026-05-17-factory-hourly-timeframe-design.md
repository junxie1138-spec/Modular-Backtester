# Factory hourly-timeframe migration — design

**Date:** 2026-05-17
**Status:** Approved (brainstorm complete, ready for implementation planning)

## Problem

The strategy factory generates and evaluates strategies exclusively on daily
(`1d`) bars. We want to know whether moving to an hourly (`1h`) intraday
timeframe produces better strategies — and, if it does, run the factory on
hourly permanently.

Two obstacles, both already researched and settled:

1. **Annualisation.** `TRADING_DAYS_PER_YEAR = 252` is hardcoded in
   `backtester/analytics/metrics.py` plus three literal `252`s in
   `wfo/multi_stitcher.py` and `engine/multi_portfolio.py`. A `timeframe`
   field is already plumbed through config → engine → WFO → strategy base
   (default `"1d"`), but the metrics layer ignores it.

2. **Data depth.** No free provider offers a decade of hourly history:
   - yfinance `1h` has a hard 730-day retention limit (not paginatable).
   - Alpha Vantage's free tier no longer serves intraday at all.
   - Polygon / Alpha Vantage premium would (paid).

   The chosen route: a **Kaggle CSV as a deep-history donor**, stitched with
   yfinance `1h` for the recent/live half — free, clears the depth wall, at
   the cost of a seam-validation step and unverified donor provenance.

## Goal

Deliver one loader architecture that serves both phases:

- **Phase A — research backfill.** Build a deep hourly dataset, run a bounded
  factory batch on it, and decide whether hourly generation beats daily.
- **Phase B — permanent feed.** If Phase A clears the gate, the same code
  becomes the factory's live hourly feed with no rewrite — the transition is
  a scheduling change, not a code change.

## Approach

**Materialised build script** (chosen over an on-the-fly `source: "stitched"`
loader). An offline script stitches donor + yfinance once and writes plain
hourly OHLCV CSVs. The backtester loader is untouched — it stays the dumb CSV
reader. The expensive seam validation runs once per build, not once per
backtest (the factory runs hundreds of backtests per batch). Donor data stays
auditable on disk. The Phase A → Phase B transition is just scheduling the
build script.

## Architecture & components

### New module: `backtester/data/hourly_stitch.py`

Pure, network-free, unit-testable functions. Lives in `backtester/data/`
because that is where the data contract and existing loaders live.

- `load_donor(path)` — read and normalize a Kaggle donor CSV to the OHLCV
  contract (`docs/data_contract.md`).
- `validate_seam(donor, recent)` — run the five seam checks (below) on the
  overlap window; return a verdict, the robust scale factor, and a coverage
  report. Makes no tradability claim.
- `splice(donor, recent, seam_ts, scale)` — cut donor strictly before the
  seam, take yfinance from the seam onward, return one clean hourly frame.

### New script: `scripts/build_hourly_dataset.py`

The CLI orchestrator — the only component that touches network and
filesystem. For each symbol:

1. If `data/donor_hourly/{SYMBOL}.csv` exists → load donor, fetch yfinance
   `1h`, `validate_seam`, `splice`, write output.
2. If no donor → yfinance `1h` only (thin, ~2yr), write output, flag.
3. If seam validation fails → fall back to yfinance-only for that symbol,
   flag loudly. The batch continues; a failure is never batch-wide.

Idempotent: re-running refreshes the yfinance tail and re-validates the seam.
That idempotency is what makes Phase B "just schedule this script."

### Data layout

| Path | Role |
|------|------|
| `data/donor_hourly/{SYMBOL}.csv` | Kaggle donor CSVs, manually placed. Provenance is unverified — a human eyeballs each file before dropping it in. |
| `data/raw_hourly/{SYMBOL}.csv` | Built output. Same OHLCV schema as `data/raw/`, hourly index. |
| `data/raw_hourly/_build_report.json` | Per-symbol audit trail (below). |

The backtester loader is **unchanged** — `data/raw_hourly/{SYMBOL}.csv` is
read by the existing `CSVDataLoader`. The factory simply repoints `root`.

### Build report

`_build_report.json`, per symbol: `source` (`stitched` / `yfinance_only` /
`failed`), donor date range, seam date, robust scale factor, validation
verdict, bar count, date span, and the tradability classification
(`tradable` / `insufficient_history`).

## Seam stitch & validation

The seam is the overlap window where both donor and yfinance `1h` have bars.
yfinance `1h` covers roughly the last 730 days; the donor must reach into
that window for a stitch to be possible. `validate_seam` runs five checks, in
order:

1. **Overlap exists.** Donor's max date must reach into yfinance's covered
   window. No overlap → abort that symbol (cannot validate a blind join).

2. **Timezone & session.** yfinance `1h` returns tz-aware US/Eastern,
   regular session only (`prepost=False`) — 7 bars/day stamped at the bar
   *open* (09:30, 10:30, … 15:30). The donor is normalized the same way: tz
   stripped to naive US/Eastern, filtered to the regular session. If the
   donor shows ~13+ bars/day it includes extended hours — filter it down; if
   it cannot be cleanly filtered, abort.

3. **Timestamp convention.** Confirm donor bars are open-stamped, not
   close-stamped, by comparing the donor's intraday timestamp set against
   yfinance's (09:30 start). A consistent 1-hour offset → shift it; an
   irregular mismatch → abort.

4. **Adjustment scale — robust estimate.** yfinance `auto_adjust=True` gives
   split + dividend-adjusted prices; the donor's adjustment basis is unknown.
   Compute the per-bar ratio `donor_close / yfinance_close` across the
   overlap. `scale` = the **median** of those ratios (not the mean — one bad
   bar must not poison the verdict). Dispersion gate: after applying the
   median scale, compute the **95th-percentile absolute relative error**
   across the overlap (equivalently, a MAD check on the ratio distribution).
   Abort only if that robust statistic exceeds tolerance. If it passes, the
   median `scale` is applied to the entire donor history, aligning it to
   yfinance's adjustment basis.

5. **Post-scale price agreement — robust.** After scaling, donor and
   yfinance closes in the overlap must agree. Judged on the 95th-percentile
   absolute relative error (~0.5% tolerance), not the worst bar.

**Splice.** Seam timestamp = the first yfinance bar. Donor supplies
everything strictly before it; yfinance supplies everything from it onward.
The live edge is therefore always pure yfinance — which keeps Phase B fresh.
Output is checked for monotonic index, no duplicate timestamps, no oversized
gap at the seam.

**Failure policy.** Per-symbol, never batch-wide. A failed symbol falls back
to yfinance-only and is flagged in `_build_report.json`. The operator reads
the report and decides.

## Minimum-history policy — outside the stitcher

`hourly_stitch.py` stays pure: it produces a frame and *reports* coverage
(bar count, date span, source). It makes **no** tradability claim.

A separate policy layer — a `min_hourly_bars` threshold checked in
`build_hourly_dataset.py` — classifies each built symbol in
`_build_report.json` as `tradable` or `insufficient_history`. A yfinance-only
fallback with ~2yr of thin data is **not** automatically tradable; if it is
below threshold it is flagged `insufficient_history`. The factory wiring must
respect this classification (below) rather than assuming any built CSV is
fair game. Stitcher reports; policy gates; the two concerns are separate.

## Annualisation fix

Single source of truth in `backtester/analytics/metrics.py`:

```python
PERIODS_PER_YEAR = {"1d": 252, "1h": 1638}
```

`1638 = 252 × 6.5` regular-session hours — documented as an approximation,
tunable. A `periods_per_year(timeframe)` accessor resolves through the map.
The metrics functions (annualized return / vol / sharpe / sortino) and the
three literal `252`s in `wfo/multi_stitcher.py` and `engine/multi_portfolio.py`
take `timeframe` (already plumbed through config → engine → WFO → strategy
base, default `"1d"`) and resolve through the accessor. Daily behaviour is
unchanged — `"1d"` still yields 252.

## Factory wiring

- **Prompt template** (`factory/prompt.py`, the embedded `data:`/`wfo:`
  block): `timeframe: "1d"` → `"1h"`; `root: "data/raw"` →
  `"data/raw_hourly"`; `start`/`end` updated to SPY's guaranteed hourly
  coverage window.

- **WFO bar counts.** WFO counts in bars (timeframe-agnostic). Current
  `train_bars: 756, test_bars: 252, step_bars: 252` (3yr / 1yr / 1yr in daily
  bars). To hold the *calendar* windows constant at hourly (~7 bars/day):
  roughly `train_bars: 5292, test_bars: 1764, step_bars: 1764`. Strategy
  logic is calendar-based, so constant calendar duration is correct.
  Consequence: one WFO window needs ~7,000 bars ≈ 4 calendar years, and
  multiple folds push higher — SPY needs ~6–8 years of hourly history. This
  is why the donor is mandatory for SPY, not optional.

- **Tradability gate.** A preflight check folded into the existing
  `factory/scripts/preflight.py` reads `_build_report.json` and **refuses to
  start a batch** if SPY is not `tradable`. Promotion **skips** any of
  AAPL / QQQ / DIA classified `insufficient_history` rather than promoting on
  thin data. ^VIX, if the VIX regime filter is active, gets the same
  treatment as an aux symbol. The factory consults the report's
  classification; it never just trusts that a CSV exists.

## Symbol scope

The build covers the full set of symbols the factory touches (~19), with
best-effort donor coverage:

- **SPY** — the factory's generation symbol. Donor mandatory.
- **14 universe names** — TSLA, NVDA, AMD, COIN, GOOGL, MSTR, XPEV, NIO,
  PLTR, SMCI, SHOP, W, META, NFLX.
- **3 promotion tickers** — AAPL, QQQ, DIA.
- **^VIX** — aux symbol for the VIX regime filter.

Donor depth is symbol-dependent: SPY / AAPL / QQQ / DIA / NVDA / META / NFLX
have a decade-plus of intraday history worth stitching; COIN (IPO 2021),
PLTR / XPEV (2020), MSTR, SMCI barely predate yfinance's own 730-day window —
for those the donor adds little and `insufficient_history` is the likely
classification. That is expected and handled by the policy layer.

## Phasing

**Phase A — research backfill.** Source donor CSVs for as many of the 19
symbols as have clean Kaggle data (SPY mandatory). Run
`build_hourly_dataset.py`. Flip the prompt template to hourly. Run a
**bounded** factory batch (`max_cycles` ≈ 50–100). Decision gate: compare the
OOS-Sortino distribution of hourly-generated strategies against the existing
daily `results.json` corpus — does hourly generation produce a
comparable-or-better hit rate above the alert / promotion thresholds?

**Phase B — permanent feed.** If Phase A clears the gate: schedule
`build_hourly_dataset.py` (cron, or a pre-batch hook in the factory loop) to
extend the live edge, set `max_cycles = 0`, leave the factory on hourly.
Phase A and Phase B share 100% of the loader code — the only difference is
operational (bounded vs continuous + scheduled rebuild).

## Testing

- **Unit — `validate_seam`:** synthetic donor/yfinance frames covering each
  verdict path — constant-ratio pass, drifting-ratio abort, single-outlier-bar
  tolerated (robust median / 95th-pct), no-overlap abort, tz mismatch,
  session mismatch (extended hours), timestamp-convention offset.
- **Unit — `splice`:** correct cut at the seam, no duplicate timestamps,
  monotonic index, no oversized gap.
- **Unit — `periods_per_year`:** map resolves `1d`→252, `1h`→1638.
- **Unit — metrics annualisation:** `timeframe="1h"` scales by √1638;
  `timeframe="1d"` unchanged at 252.
- **Unit — build policy:** a thin yfinance-only frame below `min_hourly_bars`
  is classified `insufficient_history`; a deep stitched frame is `tradable`.
- **Network paths** (`build_hourly_dataset.py`'s yfinance fetch) — not
  unit-tested; mocked via the existing `_yfinance_download` indirection seam.
- **Regression:** the existing 140 tests must stay green — the annualisation
  change must leave daily behaviour identical.

## Risks & open items

- **Donor sourcing is manual.** The operator downloads Kaggle CSVs into
  `data/donor_hourly/`. Provenance is unverified; a human eyeballs each donor
  before placing it. `_build_report.json` is the audit trail. Finding a
  single clean-provenance dataset covering all 19 tickers is unlikely —
  best-effort per-symbol coverage is the design's answer.
- **`1638` periods/year for `1h`** is an approximation (regular-session hour
  count; the 7th bar of each day is a half-length 15:30–16:00 bar). Adequate
  for v1; tunable.
- **Recent-IPO names** will mostly land as `insufficient_history` — expected,
  not a defect.
