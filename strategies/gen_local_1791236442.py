from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regime_window: int = 24
    impulse_window: int = 3
    correlation_threshold: float = 0.15
    impulse_threshold: float = 0.8
    atr_window: int = 14
    trail_atr: float = 1.8
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791236442"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.regime_window + 2, params.impulse_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.regime_window < 3 or params.impulse_window < 1 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; regime_window must be at least three.")
        if not 0.0 < params.correlation_threshold < 1.0:
            raise ValueError("correlation_threshold must be between zero and one.")
        if params.impulse_threshold <= 0.0 or params.trail_atr <= 0.0:
            raise ValueError("Impulse and trailing thresholds must be positive.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        opening = data["open"].where(data["open"] > 0.0)
        out["return"] = close.pct_change(fill_method=None)
        out["lagged_return"] = out["return"].shift(1)
        out["serial_correlation"] = out["return"].rolling(params.regime_window).corr(out["lagged_return"])
        out["return_volatility"] = out["return"].rolling(params.regime_window).std(ddof=0)
        out["impulse"] = out["return"].rolling(params.impulse_window).sum()
        denominator = out["return_volatility"] * np.sqrt(float(params.impulse_window))
        out["normalized_impulse"] = out["impulse"] / denominator.where(denominator > 0.0)
        out["candle_return"] = close / opening - 1.0
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["regime"] = np.select([
            out["serial_correlation"] >= params.correlation_threshold,
            out["serial_correlation"] <= -params.correlation_threshold,
        ], [1, -1], default=0).astype(int)
        out["desired_direction"] = np.sign(out["normalized_impulse"]) * out["regime"]
        valid = (
            np.isfinite(out["normalized_impulse"])
            & np.isfinite(out["serial_correlation"])
            & np.isfinite(out["candle_return"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0.0)
        )
        agrees = np.sign(out["candle_return"]) == out["desired_direction"]
        qualifies = valid & agrees & (out["normalized_impulse"].abs() >= params.impulse_threshold)
        out["entry_direction"] = out["desired_direction"].where(qualifies, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        high_water = np.nan
        low_water = np.nan
        stop = np.nan

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if position:
                if not valid:
                    position = 0
                    continue
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.holding_bars:
                    position = 0
                    continue
            elif valid and entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
                high_water = close
                low_water = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position

        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    close = np.array([100.0, 102.0, 103.0, 100.0, 100.0, 98.0, 97.0, 100.0])
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry_direction": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1]
    assert (result["size"] > 0.0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_indicators = strategy.indicators(flat, params)
    assert (flat_indicators["entry_direction"] == 0).all()
