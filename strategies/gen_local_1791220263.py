from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    breakout_bars: int = 20
    confirm_bars: int = 3
    atr_bars: int = 14
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791220263"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.atr_bars + 1,
                       params.breakout_bars + params.confirm_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["record_boundary"] = close.shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        out["record_close"] = (
            np.isfinite(close) & np.isfinite(out["record_boundary"])
            & close.gt(out["record_boundary"])
        )
        positions = np.arange(len(data), dtype=int)
        last_failure = np.maximum.accumulate(
            np.where(out["record_close"].to_numpy(), -1, positions)
        )
        out["record_streak"] = np.minimum(
            positions - last_failure, params.confirm_bars
        )
        out["entry"] = (
            out["record_streak"].eq(params.confirm_bars)
            & out["record_streak"].shift(1).fillna(0).lt(params.confirm_bars)
            & np.isfinite(out["ma200"])
            & close.gt(out["ma200"])
            & np.isfinite(out["atr"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if active:
                held += 1
                if np.isfinite(close):
                    # Only executed holding bars contribute to the high-water mark.
                    high_water = max(high_water, close)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, high_water - params.trail_atr * atr)
                stopped = np.isfinite(close) and close <= stop
                if stopped or held >= params.max_hold:
                    active = False
            elif entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = np.array([100.0] * 200 + [101, 102, 103, 104, 105, 50, 50])
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(len(close), 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[200:].tolist() == [0, 0, 0, 1, 1, 1, 0]
    params.max_hold = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[200:].tolist() == [0, 0, 0, 1, 1, 0, 0]
