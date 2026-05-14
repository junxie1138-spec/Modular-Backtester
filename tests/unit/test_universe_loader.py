from pathlib import Path

import pytest


def _write_universe(tmp_path, content: str) -> Path:
    p = tmp_path / "universe.yaml"
    p.write_text(content)
    return p


def test_load_universe_yaml_with_per_name_overrides(tmp_path):
    from backtester.config.universe import load_universe_config

    path = _write_universe(tmp_path,
        "universe:\n"
        "  TSLA: {sector: Auto, overrides: {entry_atr_mult: 1.5}}\n"
        "  NVDA: {sector: Semis}\n"
    )
    universe = load_universe_config(
        path=path,
        global_params={"entry_atr_mult": 1.25, "mean_lookback": 10},
    )
    assert universe["TSLA"].sector == "Auto"
    assert universe["TSLA"].effective_params["entry_atr_mult"] == 1.5
    assert universe["TSLA"].effective_params["mean_lookback"] == 10
    assert universe["NVDA"].sector == "Semis"
    assert universe["NVDA"].effective_params["entry_atr_mult"] == 1.25


def test_overrides_keys_must_be_subset_of_strategy_params(tmp_path):
    from backtester.config.universe import load_universe_config
    from backtester.core.exceptions import ConfigError

    path = _write_universe(tmp_path,
        "universe:\n"
        "  TSLA: {sector: Auto, overrides: {bogus_key: 1.5}}\n"
    )
    with pytest.raises(ConfigError, match="overrides"):
        load_universe_config(
            path=path,
            global_params={"entry_atr_mult": 1.25},
        )


def test_inline_sector_overrides_sector_map_csv(tmp_path):
    """universe.yaml's inline `sector` field wins over sector_map.csv."""
    from backtester.config.universe import load_universe_config

    # NVDA is "Semis" in sector_map.csv; we override to "Custom".
    path = _write_universe(tmp_path,
        "universe:\n"
        "  NVDA: {sector: Custom}\n"
    )
    universe = load_universe_config(path=path, global_params={})
    assert universe["NVDA"].sector == "Custom"


def test_missing_sector_raises_config_error(tmp_path):
    from backtester.config.universe import load_universe_config
    from backtester.core.exceptions import ConfigError

    path = _write_universe(tmp_path,
        "universe:\n"
        "  ZZZZ: {}\n"  # not in sector_map.csv and no inline sector
    )
    with pytest.raises(ConfigError, match="sector"):
        load_universe_config(path=path, global_params={})
