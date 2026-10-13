from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_lookback: int = 63
    compression_bars: int = 5
    atr_period: int = 14
    min_depth: float = 0.025
    max_depth: float = 0.12
    compression_atr: float = 2.0
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local-w2_1790983512"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    @staticmethod
    def warmup_bars(params: ShelfParams) -> int:
        return int(max(params.peak_lookback, params.compression_bars, params.atr_period + 1) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["peak"] = data["close"].rolling(params.peak_lookback).max()
        out["depth"] = 1.0 - data["close"] / out["peak"].replace(0.0, np.nan)
        out["floor"] = data["low"].rolling(params.compression_bars).min()
        out["ceiling"] = data["high"].rolling(params.compression_bars).max()
        out["midpoint"] = (out["floor"] + out["ceiling"]) / 2.0
        out["width_atr"] = (out["ceiling"] - out["floor"]) / out["atr"].replace(0.0, np.nan)
        out["shelf"] = (
            out["depth"].between(params.min_depth, params.max_depth)
            & out["width_atr"].le(params.compression_atr)
            & out["atr"].gt(0.0)
        ).fillna(False)
        # Both confirmation bars use the same shelf, frozen two bars earlier.
        out["entry"] = (
            out["shelf"].shift(2, fill_value=False)
            & data["close"].shift(1).gt(out["midpoint"].shift(2))
            & data["close"].gt(out["midpoint"].shift(2))
            & data["low"].shift(1).ge(out["floor"].shift(2))
            & data["low"].ge(out["floor"].shift(2))
            & data["close"].shift(1).gt(data["close"].shift(2))
            & data["close"].gt(data["close"].shift(1))
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShelfParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Fill at today's open; volatility was known at yesterday's close.
                    fixed_stop = opening[i] - params.stop_atr * atr[i - 1]
                expired = i - entry_bar + 1 >= params.hold_bars
                if not np.isfinite(close[i]) or close[i] < fixed_stop or expired:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    p = ShelfParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 110.0, "high": 111.0, "low": 109.0,
        "close": 110.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 2.0, "entry": False}, index=index)
    ind.loc[index[2], "entry"] = True
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 0, 0] + [1] * 8 + [0]
    data.loc[index[2], "close"] = 100.0
    data.loc[index[3], "close"] = 108.0
    data.loc[index[4], "close"] = 105.0
    ind.loc[index[3]:, "atr"] = 20.0
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1] + [0] * 7
    p = ShelfParams(peak_lookback=6, compression_bars=3, atr_period=3)
    close = pd.Series([100.0] * 3 + [94.0] * 3 + [94.2, 94.4, 94.6])
    sample = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": 1000.0,
    })
    sample.index = pd.date_range("2020-01-01", periods=len(sample))
    actual = GeneratedStrategy.indicators(sample, p)
    assert not actual["entry"].iloc[6] and actual["entry"].iloc[7]


if __name__ == "__main__":
    _self_check()
