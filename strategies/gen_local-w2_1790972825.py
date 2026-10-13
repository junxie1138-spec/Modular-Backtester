from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRejectionParams:
    lookback: int = 20
    atr_period: int = 14
    expansion: float = 1.6
    extension_share: float = 0.30
    breakeven_pct: float = 0.004
    trail_atr: float = 0.65
    initial_stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[RangeRejectionParams]):
    strategy_id = "gen_local-w2_1790972825"

    @classmethod
    def params_type(cls):
        return RangeRejectionParams

    def warmup_bars(self, params: RangeRejectionParams) -> int:
        return int(max(params.lookback, params.atr_period) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeRejectionParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 2:
            raise ValueError("Lookbacks must be at least two.")
        values = (params.expansion, params.extension_share, params.breakeven_pct,
                  params.trail_atr, params.initial_stop_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")
        if (params.expansion <= 1 or not 0 < params.extension_share < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.initial_stop_atr <= 0):
            raise ValueError("Invalid range or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_baseline"] = out["range"].shift(1).rolling(params.lookback).median()
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["expansion"] = out["range"] / out["range_baseline"].replace(0, np.nan)
        out["lower_extension"] = (out["previous_low"] - data["low"]).clip(lower=0)
        out["upper_extension"] = (data["high"] - out["previous_high"]).clip(lower=0)
        denominator = out["range"].replace(0, np.nan)
        out["lower_share"] = out["lower_extension"] / denominator
        out["upper_share"] = out["upper_extension"] / denominator
        out["lower_wick"] = (data[["open", "close"]].min(axis=1) - data["low"]) / denominator
        out["upper_wick"] = (data["high"] - data[["open", "close"]].max(axis=1)) / denominator
        out["recovery_up"] = (data["close"] - data["low"]) / denominator
        out["recovery_down"] = (data["high"] - data["close"]) / denominator
        out["reentered"] = ((data["close"] > out["previous_low"])
                            & (data["close"] < out["previous_high"]))
        ready = (out["expansion"] >= params.expansion) & (out["atr"] > 0) & out["reentered"]
        out["long_setup"] = (ready & (out["lower_share"] >= params.extension_share)
                             & (out["lower_extension"] > out["upper_extension"])
                             & (out["lower_wick"] >= params.extension_share))
        out["short_setup"] = (ready & (out["upper_share"] >= params.extension_share)
                              & (out["upper_extension"] > out["lower_extension"])
                              & (out["upper_wick"] >= params.extension_share))
        out["candidate"] = out["long_setup"].astype(int) - out["short_setup"].astype(int)
        out["rejection_strength"] = np.where(
            out["candidate"] > 0, out["lower_share"] * out["recovery_up"],
            np.where(out["candidate"] < 0, out["upper_share"] * out["recovery_down"], 0.0))
        out["conviction"] = ((out["expansion"] / params.expansion)
                              * out["rejection_strength"]).clip(0, 1)
        out["entry_size"] = 0.25 + 0.75 * out["conviction"].fillna(0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeRejectionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        side, age = 0, 0
        entry, peak, stop, held_size = np.nan, 0.0, 0.0, 1.0
        armed = False
        warmup = max(self.warmup_bars(params), ctx.warmup_bars)
        for i in range(warmup, len(data)):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and min(prices[i]) > 0 and high >= low
            if not valid or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                side, age, entry = 0, 0, np.nan
                continue
            if side == 0:
                if candidates[i] != 0:
                    side = int(candidates[i])
                    held_size = float(np.clip(entry_sizes[i], 0.25, 1.0))
                    entry, age, armed = np.nan, 0, False
                    signals[i], sizes[i] = side, held_size
                continue
            if np.isnan(entry):
                # The preceding close decision enters at this session's open.
                entry = opening
                peak = 0.0
                stop = -params.initial_stop_atr * atrs[i - 1]
            age += 1
            favorable = high - entry if side > 0 else entry - low
            adverse = low - entry if side > 0 else entry - high
            # Stop coordinates are profit distances: only increase for either side.
            # Test the pre-existing stop before updating from today's excursion.
            exited = adverse <= stop
            if not exited:
                peak = max(peak, favorable)
                armed = armed or peak >= entry * params.breakeven_pct
                if armed:
                    stop = max(stop, 0.0, peak - params.trail_atr * atrs[i])
                    # Only the close can breach a newly raised stop on this bar.
                    exited = side * (close - entry) <= stop
            if exited or age >= 2:
                side, age, entry = 0, 0, np.nan
            else:
                signals[i], sizes[i] = side, held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeRejectionParams(lookback=2, atr_period=2)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=3)
    ind = strategy.indicators(data, params)
    assert np.isfinite(ind.to_numpy(dtype=float)).all()
    ind["candidate"], ind["atr"], ind["entry_size"] = 0, 1.0, 0.6
    ind.loc[index[3], "candidate"] = 1
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert result["size"].iloc[4] == 0.6
    data.loc[index[4], ["high", "low", "close"]] = [101.0, 99.8, 100.2]
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[4] == 1 and result["signal"].iloc[5] == 0
    ind.loc[index[3], "candidate"] = -1
    data.loc[index[4], ["high", "low", "close"]] = [100.2, 99.0, 99.8]
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[4] == -1 and result["signal"].iloc[5] == 0
    assert set(result["signal"].unique()) <= {-1, 0, 1}
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
