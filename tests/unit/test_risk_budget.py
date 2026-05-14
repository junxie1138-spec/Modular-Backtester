import pytest


def test_risk_budget_admits_entry_below_cap():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=3_000.0,
        proposed_risk_dollars=2_000.0,
    )
    assert decision.admitted is True
    assert decision.scaled_target == 1.0


def test_risk_budget_rejects_entry_above_cap():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=5_000.0,
        proposed_risk_dollars=2_000.0,
    )
    assert decision.admitted is False
    assert decision.scaled_target == 0.0


def test_risk_budget_zero_equity_zero_admit():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=0.0, current_risk_dollars=0.0, proposed_risk_dollars=100.0,
    )
    assert decision.admitted is False


def test_risk_budget_at_exact_cap_admits():
    from backtester.engine.risk_budget import RiskBudgetEnforcer
    enforcer = RiskBudgetEnforcer(budget_pct=0.06)
    decision = enforcer.evaluate(
        portfolio_equity=100_000.0,
        current_risk_dollars=4_000.0,
        proposed_risk_dollars=2_000.0,  # exactly hits 6%
    )
    assert decision.admitted is True
