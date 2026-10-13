from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendParams:
    span: int = 8
    noise_window: int = 20
    atr_window: int = 14
    min_strength: float = 0.15
    min_innovation: float = 0.035
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[TrendParams]):
    strategy_id = "gen_local-w2_1791253010"

    @classmethod
    def params_type(cls):
        return TrendParams

    def warmup_bars(self, params: TrendParams) -> int:
        return int(max(2 * params.span + params.noise_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
        if (params.span < 2 or params.noise_window < 2 or
                params.atr_window < 1 or not 3 <= params.hold_bars <= 5 or
                not np.isfinite([params.min_strength, params.min_innovation,
                                 params.trail_atr]).all() or
                params.min_strength <= 0 or params.min_innovation <= 0 or
                params.trail_atr <= 0):
            raise ValueError("Invalid trend parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["smooth1"] = out["log_close"].ewm(span=params.span, adjust=False).mean()
        out["smooth2"] = out["smooth1"].ewm(span=params.span, adjust=False).mean()
        alpha = 2.0 / (params.span + 1.0)
        out["level"] = 2.0 * out["smooth1"] - out["smooth2"]
        out["slope"] = alpha / (1.0 - alpha) * (out["smooth1"] - out["smooth2"])
        out["slope_innovation"] = out["slope"].diff()
        out["forecast"] = (out["level"] + out["slope"]).shift(1)
        out["residual"] = out["log_close"] - out["forecast"]
        out["residual_square"] = out["residual"].pow(2)
        out["noise"] = np.sqrt(out["residual_square"].rolling(
            params.noise_window, min_periods=params.noise_window).mean()).replace(0, np.nan)
        out["strength"] = out["slope"] / out["noise"]
        out["innovation_score"] = out["slope_innovation"] / out["noise"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = (
            np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
            & (close > 0) & (out["atr"] > 0)
            & out[["strength", "innovation_score"]].notna().all(axis=1)
            & (np.arange(len(data)) >= self.warmup_bars(params))
        )
        bullish = (out["strength"] > params.min_strength) & (
            out["innovation_score"] > params.min_innovation)
        bearish = (out["strength"] < -params.min_strength) & (
            out["innovation_score"] < -params.min_innovation)
        out["candidate"] = np.select([ready & bullish, ready & bearish], [1, -1], default=0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TrendParams
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if side:
                age += 1
                if not valid:
                    side = 0
                    continue
                # Watermarks begin on the first held bar, after the entry decision.
                if side == 1:
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    proposed = high_water - params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else max(stop, proposed)
                    stopped = close <= stop
                else:
                    # The short-side mirror tracks the lowest held close.
                    low_water = close if np.isnan(low_water) else min(low_water, close)
                    proposed = low_water + params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else min(stop, proposed)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    side = 0
                signals[i] = side
                continue
            if valid and entries[i]:
                side = int(entries[i])
                age = 0
                high_water = low_water = stop = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TrendParams()
    index = pd.date_range("2020-01-01", periods=10)
    close = np.array([100, 101, 102, 103, 101, 100, 99, 98, 101, 101], dtype=float)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "candidate": [1, 1, 0, 0, 0, -1, -1, 0, 0, 0],
        "atr": [1, 1, 1, 1, 5, 1, 1, 1, 1, 1]
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, 0]
    indicators["candidate"] = 1
    indicators["atr"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist()[:7] == [0, 0, 1, 1, 1, 1, 0]
    assert strategy.indicators(data, params)["candidate"].eq(0).all()
