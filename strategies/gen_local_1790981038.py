from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTrendParams:
    trend_window: int = 8
    percentile_window: int = 126
    atr_window: int = 14
    gap_quantile: float = 0.80
    strength_quantile: float = 0.70
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2
    max_hold_days: int = 2

    def __post_init__(self):
        if min(self.trend_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < self.gap_quantile < 1 and 0 < self.strength_quantile < 1):
            raise ValueError("Quantiles must lie strictly between zero and one.")
        if not (self.breakeven_pct > 0 and self.trail_atr > 0):
            raise ValueError("Stop parameters must be positive.")
        if self.max_hold_days not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")


class GeneratedStrategy(BaseStrategy[GapTrendParams]):
    strategy_id = "gen_local_1790981038"

    @classmethod
    def params_type(cls):
        return GapTrendParams

    def warmup_bars(self, params):
        return int(params.percentile_window + max(params.trend_window, params.atr_window) + 2)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        day_range = (data["high"] - data["low"]).clip(lower=0)
        out["true_range"] = pd.concat(
            [day_range, (data["high"] - prior_close).abs(),
             (data["low"] - prior_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["body"] = data["close"] - data["open"]
        out["range"] = day_range
        out["body_sum"] = out["body"].rolling(params.trend_window).sum()
        out["range_sum"] = day_range.rolling(params.trend_window).sum()
        out["strength"] = out["body_sum"] / out["range_sum"].replace(0, np.nan)
        out["prior_strength"] = out["strength"].shift(1)
        out["strength_threshold"] = out["strength"].shift(2).rolling(
            params.percentile_window
        ).quantile(params.strength_quantile)
        out["gap"] = (data["open"] - prior_close) / out["atr"].shift(1).replace(0, np.nan)
        out["gap_threshold"] = out["gap"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.gap_quantile)
        out["setup"] = (
            (out["gap"] > 0)
            & (out["gap"] > out["gap_threshold"])
            & (out["prior_strength"] > 0)
            & (out["prior_strength"] > out["strength_threshold"])
            & (out["body"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not (np.isfinite(opens[i]) and opens[i] > 0
                                  and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                entry = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                active, armed, age = True, False, 0
            if active:
                age += 1
                close = closes[i]
                if not (np.isfinite(close) and close > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    active = False
                    continue
                # Close-observed stops exit through the next shifted signal.
                if close <= stop:
                    active = False
                    continue
                if close >= entry * (1 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, close - params.trail_atr * atr[i])
                if close <= stop or age >= params.max_hold_days:
                    active = False
                    continue
                raw[i] = 1
            elif setups[i] and i >= self.warmup_bars(params):
                raw[i] = 1
                pending = True
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapTrendParams(trend_window=2, percentile_window=2, atr_window=2)
    n = 16
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=n)
    )
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    ind = strategy.indicators(data, params)
    assert ind.index.equals(data.index)
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    ind["atr"] = 1.0
    ind["setup"] = False
    ind.iloc[6, ind.columns.get_loc("setup")] = True
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0] * 7 + [1, 1] + [0] * 7
    assert (result["size"] > 0).all()
    data.iloc[7, data.columns.get_loc("close")] = 98.0
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist() == [0] * 7 + [1] + [0] * 8
    data.iloc[7, data.columns.get_loc("close")] = 101.0
    prefix = strategy.generate_signals(data.iloc[:8], ind.iloc[:8], ctx, params).data
    full = strategy.generate_signals(data, ind, ctx, params).data
    pd.testing.assert_frame_equal(prefix, full.iloc[:8])


if __name__ == "__main__":
    _self_check()
