from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    peak_window: int = 20
    confirmation_bars: int = 3
    min_depth: float = 0.008
    max_depth: float = 0.12
    depth_move: float = 0.004
    transport_threshold: float = 0.65
    atr_period: int = 14
    stop_atr: float = 1.8
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local-w2_1790981132"

    @classmethod
    def params_type(cls):
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return int(max(params.peak_window + params.confirmation_bars,
                       params.atr_period + 1,
                       params.confirmation_bars + 1))

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        if min(params.peak_window, params.confirmation_bars, params.atr_period) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        if not 0 <= params.min_depth < params.max_depth < 1:
            raise ValueError("Invalid drawdown bounds")
        if not 0 < params.transport_threshold <= 1:
            raise ValueError("Invalid transport threshold")
        if not np.isfinite(params.depth_move) or params.depth_move <= 0:
            raise ValueError("depth_move must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        out["depth_change"] = out["depth"].diff(params.confirmation_bars)
        out["stable_peak"] = out["peak"].eq(out["peak"].shift(params.confirmation_bars))
        out["range_center"] = (data["high"] + data["low"]) / 2.0
        out["center_step"] = out["range_center"].diff()
        out["center_travel"] = out["center_step"].abs().rolling(
            params.confirmation_bars, min_periods=params.confirmation_bars
        ).sum()
        out["center_displacement"] = out["range_center"].diff(params.confirmation_bars)
        out["transport"] = out["center_displacement"] / out["center_travel"].where(
            out["center_travel"] > 0
        )
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        depth = indicators["depth"]
        eligible = depth.between(params.min_depth, params.max_depth) & indicators["stable_peak"].fillna(False)
        long_entry = eligible & (indicators["depth_change"] <= -params.depth_move) & (
            indicators["transport"] >= params.transport_threshold
        )
        short_entry = eligible & (indicators["depth_change"] >= params.depth_move) & (
            indicators["transport"] <= -params.transport_threshold
        )
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.stop_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the long high-water stop for short positions.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.stop_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.holding_bars:
                    position = 0
                    age = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                age = 0
                extreme = close
                stop = close - position * params.stop_atr * atr
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    close = pd.Series([100, 101, 103, 102, 100, 99, 97, 98, 100, 100], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.0})
    indicators = pd.DataFrame({
        "depth": 0.03,
        "stable_peak": True,
        "depth_change": [-0.01, 0, 0, 0, 0, 0.01, 0, 0, 0, 0],
        "transport": [0.8, 0, 0, 0, 0, -0.8, 0, 0, 0, 0],
        "atr": [1, 1, 1, 10, 10, 1, 1, 10, 10, 1],
    }, index=index)
    strategy = GeneratedStrategy()
    params = MomentumParams(stop_atr=1.5, holding_bars=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = data.copy()
    flat["close"] = 100.0
    indicators["atr"] = 1.0
    indicators["depth_change"] = [-0.01] + [0.0] * 9
    indicators["transport"] = [0.8] + [0.0] * 9
    params.holding_bars = 4
    result = strategy.generate_signals(flat, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
