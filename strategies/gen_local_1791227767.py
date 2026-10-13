from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetestParams:
    atr_bars: int = 14
    gap_atr: float = 0.75
    retest_atr: float = 1.0
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRetestParams]):
    strategy_id = "gen_local_1791227767"

    @classmethod
    def params_type(cls) -> type[GapRetestParams]:
        return GapRetestParams

    def warmup_bars(self, params: GapRetestParams) -> int:
        return params.atr_bars + 3

    def indicators(self, data: pd.DataFrame, params: GapRetestParams) -> pd.DataFrame:
        if params.atr_bars < 1 or params.hold_bars < 1:
            raise ValueError("Lookback and holding period must be positive.")
        if min(params.gap_atr, params.retest_atr, params.trail_atr) <= 0:
            raise ValueError("ATR multipliers must be positive.")
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        gap = data["open"] - previous_close
        shock = (
            (atr.shift(1) > 0)
            & (gap <= -params.gap_atr * atr.shift(1))
            & (data["close"] < previous_close)
        )
        floor = data["low"].shift(2)
        band = params.retest_atr * atr.shift(3)
        first_retest = (
            (data["low"].shift(1) >= floor)
            & (data["low"].shift(1) <= floor + band)
            & (data["close"].shift(1) > data["close"].shift(2))
        )
        second_retest = (
            (data["low"] >= data["low"].shift(1))
            & (data["low"] <= floor + band)
            & (data["close"] > data["close"].shift(1))
        )
        entry = (
            shock.shift(2, fill_value=False)
            & first_retest
            & second_retest
            & (data["close"] < previous_close.shift(2))
            & (atr > 0)
        ).fillna(False)
        return pd.DataFrame({
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
            "gap": gap,
            "shock": shock,
            "floor": floor,
            "band": band,
            "first_retest": first_retest.fillna(False),
            "second_retest": second_retest.fillna(False),
            "entry": entry,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRetestParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest_close = -np.inf
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    highest_close = max(highest_close, close)
                    stop = max(stop, highest_close - params.trail_atr * atr)
                if not valid or close <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                active = True
                age = 0
                # The first in-trade close belongs to the next bar's fill.
                highest_close = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = GapRetestParams()
    index = pd.date_range("2020-01-01", periods=22)
    data = pd.DataFrame({"close": 100.0}, index=index)
    features = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    features.iloc[0, 1] = True
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 3
    data.loc[index[:6], "close"] = [100, 101, 102, 101, 98, 100]
    features.loc[index[3:5], "atr"] = 10.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1] + [0] * 17
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
