from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    min_depth: float = 0.04
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790911897"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return 84

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not 0 < params.min_depth < 1 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Require 0 < min_depth < 1 and finite stop_atr > 0.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["peak"] = close.rolling(63, min_periods=63).max()
        out["previous_peak"] = out["peak"].shift(1)
        out["depth"] = (1 - close / out["peak"].where(out["peak"] > 0)).clip(lower=0)
        out["previous_depth"] = out["depth"].shift(1)
        out["basin_depth"] = out["depth"].rolling(20, min_periods=20).max().shift(1)
        out["repair_level"] = 0.5 * out["basin_depth"]
        out["breakout_level"] = close.rolling(5, min_periods=5).max().shift(1)
        out["entry"] = (
            (out["basin_depth"] >= params.min_depth)
            & (out["previous_depth"] >= out["repair_level"])
            & (out["depth"] < out["repair_level"])
            & (out["depth"] > 0)
            & (out["peak"] == out["previous_peak"])
            & (close > out["breakout_level"])
            & (out["atr"] > 0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        stop = 0.0
        entered = 0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and desired[i - 1] == 1:
                if np.isfinite(opening[i]) and opening[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entered = i
                    stop = opening[i] - params.stop_atr * atr[i - 1]
            if active:
                # Freeze the stop at entry; exit decisions fill next bar.
                if not np.isfinite(close[i]) or close[i] <= 0 or close[i] < stop or i - entered >= 7:
                    active = False
                else:
                    desired[i] = 1
            elif entry[i] and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo() -> None:
    index = pd.date_range("2020-01-01", periods=17, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 111.0, "low": 94.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 110.0
    data.iloc[3, data.columns.get_loc("close")] = 97.0
    data.iloc[4, data.columns.get_loc("close")] = 95.0
    indicators = pd.DataFrame({"atr": 2.0, "entry": False}, index=index)
    indicators.iloc[[0, 6], indicators.columns.get_loc("entry")] = True
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=84)
    result = strategy.generate_signals(data, indicators, ctx, DrawdownParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, DrawdownParams())["entry"].any()


if __name__ == "__main__":
    demo()
