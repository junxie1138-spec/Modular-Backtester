from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownTransitParams:
    peak_bars: int = 20
    rank_bars: int = 126
    transit_bars: int = 4
    tail_quantile: float = 0.8
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownTransitParams]):
    strategy_id = "gen_local-w2_1791216393"

    @classmethod
    def params_type(cls):
        return DrawdownTransitParams

    def warmup_bars(self, params: DrawdownTransitParams) -> int:
        if min(params.peak_bars, params.rank_bars, params.transit_bars,
               params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must be between 0.5 and 1.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite.")
        return int(max(params.peak_bars + params.rank_bars + params.transit_bars + 2,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownTransitParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["peak"] = close.rolling(params.peak_bars).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0)
        history = out["depth"].shift(1).rolling(params.rank_bars)
        out["deep"] = history.quantile(params.tail_quantile)
        out["shallow"] = history.quantile(1.0 - params.tail_quantile)
        out["band_valid"] = (out["deep"] > out["shallow"]) & out["shallow"].notna()
        out["deep_touch"] = (out["band_valid"] & (out["depth"] >= out["deep"])).astype(float)
        out["shallow_touch"] = (out["band_valid"] & (out["depth"] <= out["shallow"])).astype(float)
        out["recent_deep"] = out["deep_touch"].shift(1).rolling(params.transit_bars).max()
        out["recent_shallow"] = out["shallow_touch"].shift(1).rolling(params.transit_bars).max()
        out["previous_depth"] = out["depth"].shift(1)
        out["previous_shallow"] = out["shallow"].shift(1)
        out["previous_deep"] = out["deep"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["long_entry"] = (
            out["band_valid"] & (out["recent_deep"] == 1)
            & (out["depth"] <= out["shallow"])
            & (out["previous_depth"] > out["previous_shallow"])
        )
        out["short_entry"] = (
            out["band_valid"] & (out["recent_shallow"] == 1)
            & (out["depth"] > out["deep"])
            & (out["previous_depth"] <= out["previous_deep"])
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownTransitParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(self, params)
        prices = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        pending = False
        frozen_atr, stop = np.nan, np.nan
        for i in range(len(data)):
            if position:
                if pending:
                    # The prior close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        pending = False
                        continue
                    stop = opens[i] - position * params.stop_atr * frozen_atr
                    age = 0
                    pending = False
                age += 1
                stopped = np.isfinite(prices[i]) and (
                    (position == 1 and prices[i] < stop)
                    or (position == -1 and prices[i] > stop)
                )
                if stopped or age >= params.hold_bars or not np.isfinite(prices[i]):
                    position = 0
                # No same-close reentry after an exit.
            elif np.isfinite(prices[i]) and prices[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i] != shorts[i]:
                    position = 1 if longs[i] else -1
                    frozen_atr = atr[i]
                    pending = True
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check():
    p = DrawdownTransitParams(hold_bars=3, stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100., 110., 111., 100., 100., 100.],
                         "close": [100., 109., 107., 100., 103., 100.]}, index=index)
    ind = pd.DataFrame({"atr": [1.] * 6,
                        "long_entry": [True, False, False, False, False, False],
                        "short_entry": [False, False, False, True, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    data["open"] = 100.
    data["close"] = 100.
    ind["short_entry"] = False
    result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    self_check()
