from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    box_bars: int = 5
    percentile_bars: int = 80
    compression_quantile: float = 0.25
    gap_quantile: float = 0.80
    atr_bars: int = 14
    stop_atr: float = 2.0
    max_hold: int = 8
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791343066"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.box_bars, params.atr_bars) + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["previous_atr"] = out["atr"].shift(1).where(lambda x: x > 0)
        out["upper"] = data["high"].rolling(params.box_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.box_bars).min().shift(1)
        out["midpoint"] = (out["upper"] + out["lower"]) / 2.0
        out["width"] = (out["upper"] - out["lower"]) / out["previous_close"].where(lambda x: x > 0)
        out["compression_cutoff"] = out["width"].rolling(params.percentile_bars).quantile(params.compression_quantile).shift(1)
        out["gap"] = (data["open"] - out["previous_close"]) / out["previous_atr"]
        out["absolute_gap"] = out["gap"].abs()
        out["gap_cutoff"] = out["absolute_gap"].rolling(params.percentile_bars).quantile(params.gap_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        valid = np.isfinite(indicators).all(axis=1) & np.isfinite(data[["open", "close"]]).all(axis=1)
        compressed = (indicators["width"] > 0) & (indicators["width"] <= indicators["compression_cutoff"])
        extreme = indicators["absolute_gap"] > indicators["gap_cutoff"]
        long_entry = ((indicators["gap"] < 0) & (data["open"] < indicators["lower"])
                      & (data["close"] > indicators["midpoint"]) & (data["close"] <= indicators["upper"]))
        short_entry = ((indicators["gap"] > 0) & (data["open"] > indicators["upper"])
                       & (data["close"] < indicators["midpoint"]) & (data["close"] >= indicators["lower"]))
        candidates = np.where(valid & compressed & extreme & long_entry, 1,
                              np.where(valid & compressed & extreme & short_entry, -1, 0))
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        side, entry_bar, stop = 0, -1, np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # Entry uses this open; ATR was frozen at the preceding decision close.
                    entry_price = opens[i] * (1.0 + side * params.entry_slippage_bps / 10000.0)
                    stop = entry_price - side * params.stop_atr * entry_atr
                stopped = side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.max_hold
                if stopped or expired or not np.isfinite(closes[i]) or not np.isfinite(stop):
                    side = 0
                raw[i] = side
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                entry_bar, entry_atr = i + 1, atr[i]
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [98, 100, 103, 100, 98, 98],
                         "close": [100.5, 104, 101, 97, 98, 98]}, index=index)
    data["high"] = data[["open", "close"]].max(axis=1) + 1
    data["low"] = data[["open", "close"]].min(axis=1) - 1
    data["volume"] = 1000.0
    ind = pd.DataFrame({"width": 0.02, "compression_cutoff": 0.03,
                        "absolute_gap": 0.0, "gap_cutoff": 1.0, "gap": 0.0,
                        "lower": 99.0, "upper": 101.0, "midpoint": 100.0,
                        "atr": 1.0}, index=index)
    ind.loc[index[0], ["gap", "absolute_gap"]] = [-2.0, 2.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    mirror = data.copy()
    mirror["open"], mirror["close"] = 200 - data["open"], 200 - data["close"]
    mirror["high"], mirror["low"] = 200 - data["low"], 200 - data["high"]
    reverse = ind.copy()
    reverse["gap"] = -ind["gap"]
    assert strategy.generate_signals(mirror, reverse, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, 0, 0]
    params.max_hold = 2
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    invalid = ind * np.nan
    assert not strategy.generate_signals(data, invalid, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
