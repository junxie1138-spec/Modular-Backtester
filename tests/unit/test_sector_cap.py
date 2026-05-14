def test_sector_cap_admits_when_under_cap():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    # Deployed 10k in Semis (25% of 40k), new 10k Semis entry -> 20k (40% of 50k) < 50% cap -> admit.
    decision = enforcer.evaluate(
        sector="Semis",
        deployed_per_sector={"Semis": 10_000.0, "Auto": 30_000.0},
        deployed_total=40_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is True


def test_sector_cap_rejects_when_over_cap():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    # Deployed 45% Semis, new 10% Semis -> 55% > 50% -> reject.
    decision = enforcer.evaluate(
        sector="Semis",
        deployed_per_sector={"Semis": 45_000.0, "Auto": 5_000.0},
        deployed_total=50_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is False


def test_sector_cap_new_sector_no_existing_positions():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    decision = enforcer.evaluate(
        sector="Crypto",
        deployed_per_sector={"Semis": 30_000.0},
        deployed_total=30_000.0,
        proposed_dollars=10_000.0,
    )
    assert decision.admitted is True


def test_sector_cap_empty_portfolio():
    from backtester.engine.sector_cap import SectorCapEnforcer
    enforcer = SectorCapEnforcer(cap_pct=0.5)
    decision = enforcer.evaluate(
        sector="Auto",
        deployed_per_sector={},
        deployed_total=0.0,
        proposed_dollars=10_000.0,
    )
    # No existing deployment; the proposed becomes 100% of deployed.
    # The cap applies even on first entry into a sector.
    assert decision.admitted is False
