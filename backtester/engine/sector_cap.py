from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SectorDecision:
    admitted: bool


@dataclass(slots=True)
class SectorCapEnforcer:
    """Caps per-sector deployed capital at cap_pct of total portfolio equity.

    Two evaluation modes:

    1. **Equity-relative (preferred)**: pass `portfolio_equity`. The cap is
       evaluated as (sector_dollars_after_entry) / portfolio_equity. This is
       what `MultiSymbolPortfolioSimulator` uses in live runs and matches the
       PRD's intent ("≤ 50% of deployed in any one sector" interpreted as
       "no single sector can exceed 50% of total equity").

    2. **Deployed-relative (legacy)**: omit `portfolio_equity`. Falls back to
       the v0.4.0 launch semantics — cap is evaluated as new_sector / new_total
       where new_total = deployed_total + proposed. This mode degenerates on
       an empty book (first entry into any sector looks like 100% concentration)
       and is preserved only because unit tests assume it.
    """
    cap_pct: float

    def evaluate(
        self,
        *,
        sector: str,
        deployed_per_sector: dict[str, float],
        deployed_total: float,
        proposed_dollars: float,
        portfolio_equity: Optional[float] = None,
    ) -> SectorDecision:
        new_sector_dollars = deployed_per_sector.get(sector, 0.0) + proposed_dollars

        if portfolio_equity is not None:
            # Equity-relative semantics.
            if portfolio_equity <= 0:
                return SectorDecision(admitted=False)
            sector_pct = new_sector_dollars / portfolio_equity
        else:
            # Legacy deployed-relative semantics.
            new_total = deployed_total + proposed_dollars
            if new_total <= 0:
                return SectorDecision(admitted=False)
            sector_pct = new_sector_dollars / new_total

        if sector_pct > self.cap_pct + 1e-12:
            return SectorDecision(admitted=False)
        return SectorDecision(admitted=True)
