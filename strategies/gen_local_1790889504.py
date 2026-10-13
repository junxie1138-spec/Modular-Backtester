from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRenewalParams:
    range_window: int = 12
    atr_window: int = 14
    dominance: float = 0.35
    renewal_floor: float = 0.30
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RangeRenewalParams]):
    strategy_id = "gen_local_1790889504"

    @classmethod
    def params_type(cls):
        return RangeRenewalParams

    def warmup_bars(self, params: RangeRenewalParams) -> int:
        if min(params.range_window, params.atr_window, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0 < params.dominance < 1 and 0 < params.renewal_floor <= 1):
            raise ValueError("Invalid range-renewal thresholds.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")
        return max(params.range_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: RangeRenewalParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["prior_close"] = data["close"].shift(1)
        out["upper_extension"] = (data["high"] - out["prior_high"]).clip(lower=0)
        out["lower_extension"] = (out["prior_low"] - data["low"]).clip(lower=0)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["prior_close"]).abs(),
             (data["low"] - out["prior_close"]).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["upper_mass"] = out["upper_extension"].rolling(params.range_window).sum()
        out["lower_mass"] = out["lower_extension"].rolling(params.range_window).sum()
        out["total_mass"] = out["upper_mass"] + out["lower_mass"]
        out["range_mass"] = out["true_range"].rolling(params.range_window).sum()
        out["dominance"] = (
            (out["upper_mass"] - out["lower_mass"]) / out["total_mass"].replace(0, np.nan)
        )
        out["renewal"] = out["total_mass"] / out["range_mass"].replace(0, np.nan)
        ready = np.isfinite(out["atr"]) & (out["atr"] > 0)
        ready &= np.isfinite(out["dominance"]) & np.isfinite(out["renewal"])
        out["candidate"] = np.select(
            [ready & (out["renewal"] >= params.renewal_floor)
             & (out["dominance"] >= params.dominance),
             ready & (out["renewal"] >= params.renewal_floor)
             & (out["dominance"] <= -params.dominance)],
            [1, -1], default=0
        ).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeRenewalParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, 0
        extreme, stop = np.nan, np.nan
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                position = 0
                extreme = stop = np.nan
                continue
            if position:
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entry_bar >= params.max_hold:
                    position = 0
                    extreme = stop = np.nan
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
                extreme = close[i]
                stop = close[i] - position * params.trail_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), RangeRenewalParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    index = pd.date_range("2020-01-01", periods=40)
    close = pd.Series(np.arange(100.0, 140.0), index=index)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1000.0}, index=index
    )
    ind = strategy.indicators(data, params)
    first = np.flatnonzero(ind["candidate"].to_numpy())[0]
    assert ind["confirmed"].iloc[first] == 0
    assert ind["confirmed"].iloc[first + 1] == 1
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[first + 1] == 0
    assert result["signal"].iloc[first + 2] == 1
    assert result["signal"].iloc[first + 2 + params.max_hold] == 0
    for side in (1, -1):
        prices = 100 + side * np.array([0, 1, 2, 10, 9, 8, 7, 6], dtype=float)
        sample = data.iloc[:8].copy()
        sample["close"] = prices
        test_ind = pd.DataFrame(
            {"atr": [1., 1., 1., 1., 10., 10., 10., 10.],
             "confirmed": [0, side, 0, 0, 0, 0, 0, 0]}, index=sample.index
        )
        actual = strategy.generate_signals(sample, test_ind, ctx, params).data
        assert actual["signal"].tolist() == [0, 0, side, side, side, side, side, 0]
        assert (actual["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
