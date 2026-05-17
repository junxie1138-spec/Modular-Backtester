# Backlog

Deferred ideas — considered during planning, carved out of their originating
scope, not yet built. Each entry records why it was deferred and what it
would take, so it can be picked up as its own milestone later.

---

## Hourly timeframe for the factory

**Status:** Deferred — milestone-sized, not started
**Raised:** 2026-05-17, during the factory Sortino-swap design
**Source:** `docs/superpowers/specs/2026-05-17-factory-sortino-swap-design.md` (Non-goals)

Switch the strategy factory from a daily timeframe to an hourly one.
Considered alongside the Sharpe→Sortino swap and deliberately deferred — it
is too large to ride along with an unrelated change.

**Why it is milestone-sized:**

1. **Annualisation is hardcoded to daily.** Every annualised metric
   (Sharpe, Sortino, annualised return) computes through
   `TRADING_DAYS_PER_YEAR`. On hourly bars these figures are wrong until
   the annualisation factor is parameterised by timeframe.
2. **Hourly data history is short.** yfinance serves only ~730 days of
   hourly data — far shorter than the daily history the WFO windows
   assume. Data sourcing (and likely a different provider, or stitched
   history) has to be solved before WFO on hourly bars is meaningful.

**Rough scope when picked up:** parameterise the annualisation factor by
timeframe, audit every metric that assumes daily bars, design an hourly
data-sourcing path within the history limit, and revisit WFO window sizing.
