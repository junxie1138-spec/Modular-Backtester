from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth_atr: float = 2.0
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791310821"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        return 71

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not all(np.isfinite(x) and x > 0 for x in (params.depth_atr, params.trail_atr)):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["peak"] = close.rolling(60, min_periods=60).max()
        out["drawdown"] = out["peak"] - close
        out["recent_depth"] = out["drawdown"].shift(1).rolling(10, min_periods=10).max()
        out["previous_drawdown"] = out["drawdown"].shift(1)
        out["breakout_level"] = close.shift(1).rolling(3, min_periods=3).max()
        out["entry"] = (
            (out["atr"] > 0)
            & (out["recent_depth"] >= params.depth_atr * out["atr"])
            & (out["previous_drawdown"] > 0.5 * out["recent_depth"])
            & (out["drawdown"] <= 0.5 * out["recent_depth"])
            & (close > out["breakout_level"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DrawdownParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        high_water = stop = -np.inf
        entry_bar = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                active = False
                high_water = stop = -np.inf
                continue
            if active:
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or i - entry_bar >= 9:
                    active = False
                    high_water = stop = -np.inf
            elif entry[i]:
                active = True
                entry_bar = i + 1
                # The first in-trade close is on the next bar, after the entry fill.
                high_water = stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 104.0, 103.0, 100.0, 105.0]}, index=index)
    indicators = pd.DataFrame(
        {"atr": [1.0, 1.0, 8.0, 1.0, 1.0],
         "entry": [True, False, False, False, False]}, index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result.index.equals(index) and (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=90)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = strategy.indicators(data, params)
    assert not strategy.generate_signals(data, indicators, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
