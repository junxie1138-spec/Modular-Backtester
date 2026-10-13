from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 252
    gap_percentile: float = 0.25
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791332834"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.gap_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        gap = data["open"].div(previous_close.where(previous_close > 0)).sub(1)
        monday = pd.Series(data.index.dayofweek == 0, index=data.index)
        monday_gap = gap.where(monday)
        threshold = monday_gap.rolling(
            params.gap_window, min_periods=max(2, params.gap_window // 14)
        ).quantile(params.gap_percentile).shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        entry = (
            monday
            & (gap < 0)
            & (gap < threshold)
            & (data["close"] > data["open"])
            & (data["close"] < previous_close)
            & (atr > 0)
        )
        return pd.DataFrame(
            {
                "previous_close": previous_close,
                "gap": gap,
                "monday": monday,
                "monday_gap": monday_gap,
                "gap_threshold": threshold,
                "true_range": true_range,
                "atr": atr,
                "entry": entry.fillna(False),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The entry open is observed only on the execution bar.
                    stop = opens[i] - params.stop_atr * entry_atr
                if (
                    not np.isfinite(stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"atr": 2.0, "entry": False}, index=index)
    ind.loc[index[0], "entry"] = True
    strategy = GeneratedStrategy()
    timed = strategy.generate_signals(data, ind, None, Params()).data
    assert timed["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    ind.loc[index[1]:, "atr"] = 50.0
    data.loc[index[2], "close"] = 99.0
    data.loc[index[3], "close"] = 95.0
    stopped = strategy.generate_signals(data, ind, None, Params()).data
    assert stopped["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    assert (stopped["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
