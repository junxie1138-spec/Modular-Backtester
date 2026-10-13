from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietStreakParams:
    streak_length: int = 3
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[QuietStreakParams]):
    strategy_id = "gen_local_1790869082"

    @classmethod
    def params_type(cls):
        return QuietStreakParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_period + 1, params.streak_length + 1))

    def indicators(self, data, params):
        if params.streak_length < 1 or params.atr_period < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if not np.isfinite(params.entry_slippage_bps) or params.entry_slippage_bps < 0:
            raise ValueError("entry_slippage_bps must be finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_volume"] = data["volume"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["quiet_advance"] = (
            (data["close"] > out["previous_close"])
            & (data["volume"] < out["previous_volume"])
            & (data["volume"] > 0)
        ).fillna(False)
        quiet = out["quiet_advance"]
        out["streak"] = quiet.astype(int).groupby((~quiet).cumsum()).cumsum()
        out["entry"] = (
            (out["streak"] == params.streak_length)
            & (data["close"] > out["ma200"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        decisions = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        stop = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Capture the fill bar's open only once that bar is observed.
                    entry_price = opening[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                    stop = entry_price - params.stop_atr * entry_atr
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                if active and (
                    not np.isfinite(close[i])
                    or close[i] < stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = QuietStreakParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=216, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.loc[index[198:201], "close"] = [100.5, 101.0, 101.5]
    data.loc[index[198:201], "volume"] = [900.0, 800.0, 700.0]
    derived = strategy.indicators(data, params)
    assert derived["streak"].iloc[200] == 3 and derived["entry"].iloc[200]
    assert not derived["entry"].iloc[:199].any()
    data.loc[index[201], "open"] = 103.0
    data.loc[index[202], "close"] = 98.0
    derived["entry"] = False
    derived.loc[index[[200, 204]], "entry"] = True
    derived["atr"] = 2.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, derived, ctx, params).data
    assert signals["signal"].iloc[200:204].tolist() == [0, 1, 1, 0]
    assert signals["signal"].iloc[205:213].eq(1).all()
    assert signals["signal"].iloc[213] == 0
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
