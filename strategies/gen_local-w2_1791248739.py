from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_window: int = 60
    compression_quantile: float = 0.35
    streak_window: int = 126
    streak_quantile: float = 0.90
    min_streak: int = 2
    atr_window: int = 14
    breakeven_gain: float = 0.004
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791248739"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.range_window + params.streak_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.range_window, params.streak_window, params.atr_window, params.min_streak) < 1:
            raise ValueError("Lookbacks and minimum streak must be positive.")
        if not (0 < params.compression_quantile < 1 and 0 < params.streak_quantile < 1):
            raise ValueError("Quantiles must lie between zero and one.")
        if params.breakeven_gain <= 0 or params.trail_atr <= 0:
            raise ValueError("Stop parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["range_fraction"] = (data["high"] - data["low"]) / close.where(close > 0)
        out["range_limit"] = out["range_fraction"].rolling(params.range_window).quantile(params.compression_quantile).shift(1)
        out["compressed"] = (out["range_fraction"] <= out["range_limit"]) & out["range_limit"].notna()
        groups = (~out["compressed"]).cumsum()
        out["streak"] = out["compressed"].astype(int).groupby(groups).cumsum()
        # Exclude pre-calibration zeros from the historical streak distribution.
        calibrated_streak = out["streak"].where(out["range_limit"].notna())
        out["streak_limit"] = calibrated_streak.rolling(params.streak_window).quantile(params.streak_quantile).shift(1)
        out["prior_streak"] = out["streak"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        prior_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry_ready"] = (
            (out["prior_streak"] >= params.min_streak)
            & (out["prior_streak"] >= out["streak_limit"])
            & (out["range_fraction"] > out["range_limit"])
            & (close > out["prior_high"])
            & (close > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        age = 0
        entry = peak = stop = 0.0
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    age = 0
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(highs[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                else:
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry * (1 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # All stop decisions are confirmed at the close and filled next open.
                    if closes[i] <= stop or age >= 2:
                        active = False
                decisions[i] = int(active)
            elif ready[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = CompressionParams()
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.0, 101.0, 101.0, 101.0, 101.0],
        "low": [99.0] * 5,
        "close": [100.0, 100.8, 100.8, 100.8, 100.8],
        "volume": [1000.0] * 5,
    }, index=pd.date_range("2020-01-01", periods=5))
    ind = pd.DataFrame({"atr": 1.0, "entry_ready": [True, False, False, False, False]}, index=data.index)
    signals = strategy.generate_signals(data, ind, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[data.index[1], ["high", "close"]] = [100.6, 99.9]
    signals = strategy.generate_signals(data, ind, None, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert not actual["entry_ready"].any()
    assert strategy.warmup_bars(params) >= params.range_window + params.streak_window + 1


if __name__ == "__main__":
    _self_check()
