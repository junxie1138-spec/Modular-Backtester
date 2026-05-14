from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorDecision:
    admitted: bool


@dataclass(slots=True)
class SectorCapEnforcer:
    """Caps per-sector deployed capital at cap_pct of total deployed."""
    cap_pct: float

    def evaluate(
        self,
        *,
        sector: str,
        deployed_per_sector: dict[str, float],
        deployed_total: float,
        proposed_dollars: float,
    ) -> SectorDecision:
        new_sector_dollars = deployed_per_sector.get(sector, 0.0) + proposed_dollars
        new_total = deployed_total + proposed_dollars
        if new_total <= 0:
            return SectorDecision(admitted=False)
        new_sector_pct = new_sector_dollars / new_total
        if new_sector_pct > self.cap_pct + 1e-12:
            return SectorDecision(admitted=False)
        return SectorDecision(admitted=True)
