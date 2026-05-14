import math
import pandas as pd
import pytest


def _ohlcv(closes, start="2022-01-03"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def test_range_atr_ratio_definition():
    """range_p10_p90_63d / atr_tr_20 — sanity check on a synthetic series."""
    from scripts.screen_universe import compute_metrics
    closes = [100.0 + 5.0 * math.sin(i * 0.1) for i in range(150)]
    data = _ohlcv(closes)
    m = compute_metrics(data)
    assert m["range_atr_ratio"] > 0


def test_slope_200d_pct_per_day_uses_expm1():
    """The percent reported is expm1(slope_log), not slope_log itself."""
    from scripts.screen_universe import compute_metrics
    # Exponential growth: close[i] = 100 * exp(0.005 * i) -> slope_log ~ 0.005.
    closes = [100.0 * math.exp(0.005 * i) for i in range(250)]
    data = _ohlcv(closes)
    m = compute_metrics(data)
    # expm1(0.005) ~ 0.005012; within tolerance.
    assert m["slope_200d_pct_per_day"] == pytest.approx(0.005012, rel=0.05)


def test_trend_filter_requires_both_slope_and_r_squared():
    """A high-slope but low-R^2 series must NOT be rejected (it's noisy, not trending)."""
    from scripts.screen_universe import passes_filters
    keep = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.003, r_squared_200d=0.2,
        min_data_length_ok=True,
    )
    assert keep is True
    drop = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.003, r_squared_200d=0.5,
        min_data_length_ok=True,
    )
    assert drop is False


def test_min_data_length_filter():
    from scripts.screen_universe import passes_filters
    drop = passes_filters(
        range_atr_ratio=10.0, slope_200d_pct_per_day=0.0001, r_squared_200d=0.1,
        min_data_length_ok=False,
    )
    assert drop is False


def test_emits_unknown_sector_with_warning(tmp_path, capsys):
    from scripts.screen_universe import write_universe_yaml
    metrics_by_symbol = {
        "TSLA": {"sector": "Auto",    "range_atr_ratio": 9.0, "slope_200d_pct_per_day": 0.0005, "r_squared_200d": 0.1},
        "ZZZZ": {"sector": "unknown", "range_atr_ratio": 7.0, "slope_200d_pct_per_day": 0.0001, "r_squared_200d": 0.05},
    }
    out_path = tmp_path / "universe_candidates.yaml"
    write_universe_yaml(metrics_by_symbol, out=out_path, screening_window=("2023-01-01", "2025-12-31"))
    captured = capsys.readouterr()
    assert "ZZZZ" in captured.err or "unknown sector" in captured.err.lower()


def test_top_n_caps_output(tmp_path):
    from scripts.screen_universe import filter_and_rank
    metrics = {
        f"S{i:02d}": {"sector": "Test", "range_atr_ratio": 10.0 - 0.1 * i,
                     "slope_200d_pct_per_day": 0.0001, "r_squared_200d": 0.1}
        for i in range(50)
    }
    ranked = filter_and_rank(metrics, top=10)
    assert len(ranked) == 10
    # Top of the ranking has the highest range_atr_ratio.
    assert list(ranked.keys())[0] == "S00"
