from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    atr_period: int = 14
    volume_multiple: float = 1.1
    stop_atr: float = 0.75
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790967317"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.lookback, params.atr_period) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; lookback must be at least two.")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if params.hold_days not in (1, 2):
            raise ValueError("hold_days must be one or two.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["volume_sum"] = data["volume"].rolling(params.lookback).sum().shift(1)
        out["close_volume_sum"] = (close * data["volume"]).rolling(params.lookback).sum().shift(1)
        out["centroid"] = out["close_volume_sum"] / out["volume_sum"].replace(0, np.nan)
        out["range_high"] = data["high"].rolling(params.lookback).max().shift(1)
        out["range_low"] = data["low"].rolling(params.lookback).min().shift(1)
        out["position"] = (close - out["range_low"]) / (out["range_high"] - out["range_low"]).replace(0, np.nan)
        out["volume_baseline"] = out["volume_sum"] / params.lookback
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].replace(0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["reclaim"] = (
            (close > out["centroid"])
            & (out["previous_close"] <= out["centroid"].shift(1))
        ).fillna(False)
        out["qualified"] = (
            (close > out["previous_close"])
            & (close > out["centroid"])
            & out["position"].between(0.25, 0.80)
            & (out["volume_ratio"] >= params.volume_multiple)
            & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = (
            out["qualified"]
            & out["qualified"].shift(1, fill_value=False)
            & out["reclaim"].shift(1, fill_value=False)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                # Entry fills on this bar's open or an earlier open.
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.stop_atr * atr[i])
                    if close[i] <= stop or i - entry_bar >= params.hold_days:
                        active = False
            elif valid and entries[i]:
                active = True
                entry_bar = i
                # Exclude the decision close: it precedes the entry fill.
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 100., 103., 101., 100.],
        "high": [101., 101., 104., 104., 102., 101.],
        "low": [99., 99., 99., 100., 99., 99.],
        "close": [100., 100., 103., 101., 100., 100.],
        "volume": [100.] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [0.5] * 6,
        "confirmed": [False, True, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    one_day = GeneratedStrategy.generate_signals(data, indicators, None, Params(hold_days=1)).data
    assert one_day["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    actual = GeneratedStrategy.indicators(data, Params())
    assert not actual["confirmed"].any()
    assert result.index.equals(index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
