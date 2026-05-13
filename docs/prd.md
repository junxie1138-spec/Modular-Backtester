# PRD (snapshot)

See the implementation plan at `docs/superpowers/plans/2026-05-14-modular-stock-backtester.md`
for the authoritative product requirements that drove this codebase.

Headline goals:
- New strategies are single drop-in files.
- One ABC contract; engine never special-cases a strategy.
- Same engine powers backtest / optimize / WFO.
- All runs are config-driven and write deterministic artifacts.
