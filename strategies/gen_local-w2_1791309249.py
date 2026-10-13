from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 3
    history_bars: int = 252
    calibration_bars: int = 126
    analog_percentile: float = 0.10
    entry_percentile: float = 0.80


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791309249"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if params.shape_bars < 2 or params.history_bars < 20 or params.calibration_bars < 2:
            raise ValueError("Lookbacks are too short")
        if not 0.0 < params.analog_percentile <= 1.0:
            raise ValueError("analog_percentile must be in (0, 1]")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be in (0, 1)")
        return int(2 * params.shape_bars + params.history_bars + params.calibration_bars + 2)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        result["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        result["scale"] = result["return"].pow(2).rolling(params.shape_bars).mean().pow(0.5).replace(0.0, np.nan)
        feature_columns = []
        for lag in range(params.shape_bars):
            name = f"shape_{lag}"
            result[name] = result["return"].shift(lag) / result["scale"]
            feature_columns.append(name)
        patterns = result[feature_columns].to_numpy()
        prices = close.to_numpy()
        forecast = np.full(len(data), np.nan)
        distance_cutoff = np.full(len(data), np.nan)
        analog_count = np.zeros(len(data), dtype=float)
        first = 2 * params.shape_bars + params.history_bars + 1
        # ponytail: bounded exhaustive matching; use a neighbor index if histories grow materially.
        for t in range(first, len(data)):
            if not np.isfinite(patterns[t]).all():
                continue
            # Historical two-day outcomes finish before the current shape begins.
            end = t - params.shape_bars - 2
            candidates = np.arange(end - params.history_bars + 1, end + 1)
            distance = np.mean((patterns[candidates] - patterns[t]) ** 2, axis=1)
            outcomes = prices[candidates + 2] / prices[candidates] - 1.0
            valid = np.isfinite(distance) & np.isfinite(outcomes)
            if valid.sum() < 8:
                continue
            cutoff = np.quantile(distance[valid], params.analog_percentile)
            chosen = valid & (distance <= cutoff)
            distance_cutoff[t] = cutoff
            analog_count[t] = float(chosen.sum())
            if chosen.sum() >= 8:
                forecast[t] = float(np.median(outcomes[chosen]))
        result["distance_cutoff"] = distance_cutoff
        result["analog_count"] = analog_count
        result["forecast"] = forecast
        result["entry_threshold"] = result["forecast"].shift(1).rolling(params.calibration_bars).quantile(params.entry_percentile).clip(lower=0.0)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        ready = indicators[["forecast", "entry_threshold"]].notna().all(axis=1)
        entry_condition = ready & indicators["forecast"].gt(indicators["entry_threshold"])
        # Exposure persists while the entry condition holds; its reversal is the only exit.
        df["signal"] = entry_condition.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = AnalogParams()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:500], params)
    np.testing.assert_allclose(full.iloc[:500].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert full["entry_threshold"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    signals = strategy.generate_signals(data, full, None, params).data
    expected = (full["forecast"] > full["entry_threshold"]).astype(int).shift(1).fillna(0).astype(int)
    np.testing.assert_array_equal(signals["signal"].to_numpy(), expected.to_numpy())
    assert set(signals["signal"].unique()) <= {0, 1}
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
