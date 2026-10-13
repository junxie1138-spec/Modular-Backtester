from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 12
    regime_window: int = 20
    percentile_window: int = 96
    innovation_quantile: float = 0.85
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790981795"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return max(params.slope_window + 1, params.regime_window + 1, params.atr_window) + params.percentile_window + 1

    def indicators(self, data, params):
        if min(params.slope_window, params.regime_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.5 < params.innovation_quantile < 1 or params.trail_atr <= 0 or params.breakeven_pct <= 0 or params.max_hold < 1:
            raise ValueError("Invalid strategy parameters.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["change"] = out["log_close"].diff()
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2
        out["mean_level"] = out["log_close"].rolling(n).mean()
        out["slope"] = np.nan
        if len(data) >= n:
            slopes = np.convolve(out["log_close"].to_numpy(), weights[::-1], mode="valid") / np.dot(weights, weights)
            out.iloc[n - 1:, out.columns.get_loc("slope")] = slopes
        out["latent_level"] = out["mean_level"] + out["slope"] * (n - 1) / 2
        out["slope_change"] = out["slope"].diff()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["relative_atr"] = out["atr"] / data["close"].where(data["close"] > 0)
        out["innovation"] = out["slope_change"] / out["relative_atr"].replace(0, np.nan)
        out["innovation_cutoff"] = out["innovation"].abs().shift(1).rolling(params.percentile_window).quantile(params.innovation_quantile)
        out["net_change"] = out["change"].rolling(params.regime_window).sum()
        out["gross_change"] = out["change"].abs().rolling(params.regime_window).sum()
        out["efficiency"] = out["net_change"].abs() / out["gross_change"].replace(0, np.nan)
        history = out["efficiency"].shift(1).rolling(params.percentile_window)
        out["regime_high"] = history.quantile(0.65)
        out["regime_low"] = history.quantile(0.35)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": 1.0}, index=data.index)
        columns = ["slope", "innovation", "innovation_cutoff", "efficiency", "regime_high", "regime_low", "atr"]
        values = indicators[columns].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        regime, side, age = 0, 0, 0
        entry, stop = np.nan, np.nan
        armed = False
        for i, row in enumerate(values):
            slope, innovation, cutoff, efficiency, high, low, atr = row
            valid = np.isfinite(row).all() and atr > 0 and np.isfinite(closes[i]) and closes[i] > 0
            if not valid:
                side, regime = 0, 0
                entry = np.nan
                continue
            if efficiency >= high:
                regime = 1
            elif efficiency <= low:
                regime = -1
            if side:
                if np.isnan(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    entry_atr = values[i - 1, -1] if i else atr
                    stop = side * entry - params.trail_atr * entry_atr
                    age, armed = 0, False
                age += 1
                favorable = side * closes[i]
                # Signed-price stops rise for both directions; short price stops fall.
                if favorable <= stop or age >= params.max_hold:
                    side, entry = 0, np.nan
                else:
                    if favorable - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry, favorable - params.trail_atr * atr)
                signals[i] = side
                continue
            if cutoff > 0 and abs(innovation) > cutoff and regime:
                direction = 1 if innovation > 0 else -1
                if regime == -1 or direction * slope > 0:
                    side = direction if regime == 1 else -direction
                    entry = np.nan
            signals[i] = side
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        close = np.array([100, 100 + side * 2, 100 - side, 100, 100, 100], dtype=float)
        data = pd.DataFrame({"open": 100.0, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"slope": float(side), "innovation": [side * 2, 0, 0, 0, 0, 0], "innovation_cutoff": 1.0, "efficiency": 0.8, "regime_high": 0.7, "regime_low": 0.3, "atr": 1.0}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0, 0]
        assert result["size"].gt(0).all()
    short_data = data.iloc[:2]
    assert strategy.indicators(short_data, params).index.equals(short_data.index)


if __name__ == "__main__":
    _self_check()
