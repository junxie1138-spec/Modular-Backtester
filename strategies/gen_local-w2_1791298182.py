from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 2
    accel_floor: float = 0.0025
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791298182"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.roc_bars + 2, 15)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean().fillna(np.inf)
        out["atr"] = tr.rolling(14, min_periods=14).mean().fillna(0.0)
        out["roc"] = close.pct_change(params.roc_bars).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["accel"] = out["roc"].diff().fillna(0.0)
        out["prior_accel"] = out["accel"].shift(1).fillna(0.0)
        prior_high = close.rolling(5, min_periods=5).max().shift(1)
        out["dip_atr"] = ((prior_high - prev_close) / out["atr"].shift(1).replace(0, np.nan)).fillna(0.0)
        out["entry"] = (
            (close > out["ma200"])
            & (out["atr"] > 0)
            & (out["roc"] > 0)
            & (out["accel"] >= params.accel_floor)
            & (out["prior_accel"] < 0)
            & out["dip_atr"].between(0.3, 1.2)
            & (close > prev_close)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        states = np.zeros(len(data), dtype=int)
        in_trade = False
        peak = stop = 0.0
        held = 0
        for i, (price, atr, ma, enter) in enumerate(
            zip(
                data["close"].to_numpy(),
                indicators["atr"].to_numpy(),
                indicators["ma200"].to_numpy(),
                indicators["entry"].to_numpy(),
            )
        ):
            if in_trade:
                held += 1
                peak = max(peak, price)
                stop = max(stop, peak - params.trail_atr * atr)
                if price <= stop or held >= 2 or price <= ma:
                    in_trade = False
                else:
                    states[i] = 1
            elif enter:
                in_trade = True
                peak = price
                stop = price - params.trail_atr * atr
                held = 0
                states[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = states
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame({"close": [100.0, 102.0, 100.2]}, index=pd.date_range("2020-01-01", periods=3))
    derived = pd.DataFrame(
        {"atr": [1.0, 1.0, 10.0], "ma200": [1.0] * 3, "entry": [True, False, False]},
        index=sample.index,
    )
    result = GeneratedStrategy().generate_signals(
        sample, derived, StrategyContext("SPY", "1d", 200), StrategyParams()
    )
    assert result.data["signal"].tolist() == [0, 1, 1]
