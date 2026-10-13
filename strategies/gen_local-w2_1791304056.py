from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTraverseParams:
    range_bars: int = 6
    volume_bars: int = 20
    volume_multiple: float = 1.4
    atr_bars: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeTraverseParams]):
    strategy_id = "gen_local-w2_1791304056"

    @classmethod
    def params_type(cls):
        return RangeTraverseParams

    def warmup_bars(self, params: RangeTraverseParams) -> int:
        return max(200, params.range_bars + 1, params.volume_bars + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RangeTraverseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["range_high"] = data["high"].shift(1).rolling(params.range_bars).max()
        out["range_low"] = data["low"].shift(1).rolling(params.range_bars).min()
        out["range_width"] = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = (data["close"] - out["range_low"]) / out["range_width"]
        out["previous_close"] = data["close"].shift(1)
        out["previous_position"] = (out["previous_close"] - out["range_low"]) / out["range_width"]
        out["volume_base"] = data["volume"].shift(1).rolling(params.volume_bars).median()
        out["volume_ratio"] = data["volume"] / out["volume_base"].replace(0.0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        ready = np.isfinite(out).all(axis=1) & out["atr"].gt(0.0)
        confirmed = ready & out["volume_ratio"].ge(params.volume_multiple)
        long_entry = (
            confirmed & out["previous_position"].le(0.25) & out["position"].gt(0.5)
            & data["close"].gt(out["ma200"]) & data["close"].gt(data["open"])
        )
        short_entry = (
            confirmed & out["previous_position"].ge(0.75) & out["position"].lt(0.5)
            & data["close"].lt(out["ma200"]) & data["close"].lt(data["open"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeTraverseParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2) or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("hold_bars must be 1 or 2 and stop_atr must be positive.")
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        age = 0
        entry_atr = np.nan
        stop_price = np.nan
        for i in range(len(data)):
            if position:
                if age == 0:
                    # Anchor once to the next-bar entry open and decision-bar ATR.
                    stop_price = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                stopped = (
                    not np.isfinite(stop_price)
                    or (position == 1 and closes[i] < stop_price)
                    or (position == -1 and closes[i] > stop_price)
                )
                if stopped or age >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_atr = atr[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeTraverseParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "close": [100.0, 98.0, 100.0, 100.0, 100.5, 100.0, 100.0]},
        index=index,
    )
    features = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0], "atr": 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    data.loc[index[4], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
