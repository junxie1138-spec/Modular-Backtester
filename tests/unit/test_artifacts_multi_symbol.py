def test_artifact_writer_writes_resolved_config_with_universe_expansion(tmp_path):
    from backtester.io.artifacts import ArtifactWriter
    from backtester.config.models import (
        RunConfig, DataConfig, ExecutionConfig, PortfolioConfig,
    )
    from backtester.config.universe import ResolvedSymbolConfig

    rc = RunConfig(
        run_name="vtest",
        strategy="mean_reversion_atr",
        strategy_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
        data=DataConfig(source="csv", root="data/raw",
                        start="2024-01-01", end="2024-06-30", timeframe="1d",
                        symbols=[]),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
        output_root=str(tmp_path),
        universe_path="configs/universe.yaml",
    )
    writer = ArtifactWriter(root=str(tmp_path), run_name=rc.run_name)
    universe = {
        "TSLA": ResolvedSymbolConfig(
            symbol="TSLA", sector="Auto",
            effective_params={"entry_atr_mult": 1.5, "mean_lookback": 10},
        ),
        "NVDA": ResolvedSymbolConfig(
            symbol="NVDA", sector="Semis",
            effective_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
        ),
    }
    writer.write_config(rc, resolved_universe=universe)

    import yaml
    with (writer.run_dir / "config_resolved.yaml").open() as f:
        doc = yaml.safe_load(f)
    assert "resolved_universe" in doc
    assert doc["resolved_universe"]["TSLA"]["effective_params"]["entry_atr_mult"] == 1.5
    assert doc["resolved_universe"]["NVDA"]["sector"] == "Semis"
