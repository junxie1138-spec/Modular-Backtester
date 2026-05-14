from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    admitted: bool
    scaled_target: float  # 0.0 if rejected, 1.0 if fully admitted


@dataclass(slots=True)
class RiskBudgetEnforcer:
    """Caps total portfolio risk (sum of position x stop-distance) at budget_pct of equity.

    `current_risk_dollars` is the simulator's running tally; `proposed_risk_dollars` is the
    risk a new entry would add. Decision is binary in v0.4.0 (admit-or-drop).
    """
    budget_pct: float

    def evaluate(
        self,
        *,
        portfolio_equity: float,
        current_risk_dollars: float,
        proposed_risk_dollars: float,
    ) -> RiskDecision:
        if portfolio_equity <= 0:
            return RiskDecision(admitted=False, scaled_target=0.0)
        total_risk_pct = (current_risk_dollars + proposed_risk_dollars) / portfolio_equity
        if total_risk_pct > self.budget_pct + 1e-12:
            return RiskDecision(admitted=False, scaled_target=0.0)
        return RiskDecision(admitted=True, scaled_target=1.0)
