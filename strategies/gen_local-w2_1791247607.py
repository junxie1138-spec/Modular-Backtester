from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendParams:
    trend_span: int = 10
    threshold_window: int = 63
    entry_quantile: float = 0.85
    atr_period: int = 14
    trail_atr: float = 1.8
    holding_bars: int = 4

    def __post_init__(self):
        if self.trend_span < 2 or self.threshold_window < 5 or self.atr_period < 2:
            raise ValueError("Lookbacks are too short")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        if not 3 <= self.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")


class GeneratedStrategy(BaseStrategy[TrendParams]):
    strategy_id = "gen_local-w2_1791247607"

    @classmethod
    def params_type(cls):
        return TrendParams

    def warmup_bars(self, params: TrendParams) -> int:
        return int(2 * params.trend_span + params.atr_period + params.threshold_window + 1)

    def indicators(self, data: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_close"] = np.log(close.where(close > 0))
        out["smooth_once"] = out["log_close"].rolling(params.trend_span).mean()
        out["smooth_twice"] = out["smooth_once"].rolling(params.trend_span).mean()
        # Double moving-average trend model removes the smoother's linear lag.
        out["latent_level"] = 2.0 * out["smooth_once"] - out["smooth_twice"]
        out["latent_slope"] = 2.0 * (out["smooth_once"] - out["smooth_twice"]) / (params.trend_span - 1)
        out["forecast"] = out["latent_level"] + out["latent_slope"]
        out["forecast_error"] = out["log_close"] - out["forecast"].shift(1)
        out["residual_noise"] = out["forecast_error"].abs().rolling(params.atr_period).mean()
        out["slope_innovation"] = out["latent_slope"].diff()
        out["innovation_score"] = out["slope_innovation"] / out["residual_noise"].replace(0.0, np.nan)
        out["innovation_score"] = out["innovation_score"].replace([np.inf, -np.inf], np.nan)
        out["entry_threshold"] = out["innovation_score"].abs().shift(1).rolling(params.threshold_window).quantile(params.entry_quantile)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendParams,
    ) -> SignalFrame:
        score = indicators["innovation_score"]
        threshold = indicators["entry_threshold"]
        slope = indicators["latent_slope"]
        error = indicators["forecast_error"]
        long_entry = (score > threshold) & (slope > 0) & (error > 0)
        short_entry = (score < -threshold) & (slope < 0) & (error < 0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                position, age, extreme, stop = 0, 0, np.nan, np.nan
                continue
            if position:
                # Entry executes this morning; start tracking closes only now.
                age += 1
                if position == 1:
                    extreme = close if np.isnan(extreme) else max(extreme, close)
                    candidate = extreme - params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close <= stop
                else:
                    # Mirror the high-water trailing stop for short positions.
                    extreme = close if np.isnan(extreme) else min(extreme, close)
                    candidate = extreme + params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.holding_bars:
                    position, age, extreme, stop = 0, 0, np.nan, np.nan
            elif entries[i]:
                position = int(entries[i])
                age, extreme, stop = 0, np.nan, np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = TrendParams(trail_atr=1.0, holding_bars=5)
    index = pd.date_range("2020-01-01", periods=8)
    for direction in (1, -1):
        closes = np.array([100, 100, 103, 101, 100, 100, 100, 100], dtype=float)
        if direction == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1000.0}, index=index)
        derived = pd.DataFrame({
            "innovation_score": [2 * direction] + [0] * 7,
            "entry_threshold": 1.0,
            "latent_slope": float(direction),
            "forecast_error": float(direction),
            "atr": [1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0],
        }, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0, 0]
        assert result["size"].gt(0).all()
        data["close"] = 100.0
        flat = strategy.indicators(data, params)
        assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()
    derived["atr"] = 1.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
