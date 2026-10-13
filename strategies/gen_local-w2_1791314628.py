from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_min: float = 0.025
    compression_max: float = 0.35
    breakeven_gain: float = 0.015
    initial_atr: float = 2.0
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791314628"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.drawdown_min < 1 and 0 < params.compression_max < 1
                and params.breakeven_gain > 0 and params.initial_atr > 0
                and params.trail_atr > 0 and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak40"] = data["high"].rolling(40, min_periods=40).max()
        out["depth"] = 1.0 - close / out["peak40"].replace(0, np.nan)
        out["deepest5"] = out["depth"].rolling(5, min_periods=5).max()
        out["high5"] = data["high"].rolling(5, min_periods=5).max()
        out["low5"] = data["low"].rolling(5, min_periods=5).min()
        out["prior_high20"] = data["high"].shift(5).rolling(20, min_periods=20).max()
        out["prior_low20"] = data["low"].shift(5).rolling(20, min_periods=20).min()
        out["compression"] = (out["high5"] - out["low5"]) / (
            out["prior_high20"] - out["prior_low20"]).replace(0, np.nan)
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (close > out["ma200"])
            & (out["deepest5"] >= params.drawdown_min)
            & (out["depth"] >= 0.65 * out["deepest5"])
            & (out["compression"] <= params.compression_max)
            & (close > out["previous_close"])
            & (close >= (out["high5"] + out["low5"]) / 2.0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = stop = peak = entry_atr = 0.0
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite([opening, high, low, close, atr[i], regime[i]]).all()
            if not valid or min(opening, high, low, close) <= 0 or atr[i] <= 0:
                active = pending = armed = False
                continue
            if pending:
                # Yesterday's target fills at today's open, never yesterday's close.
                entry = opening
                peak = entry
                stop = entry - params.initial_atr * entry_atr
                age = 0
                active, pending, armed = True, False, False
            if active:
                age += 1
                # Test the pre-existing stop before using today's high to ratchet.
                if low <= stop or close <= regime[i] or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, high)
                if peak >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if close <= stop:
                        active = False
                        continue
                signal[i] = 1
            elif entries[i]:
                pending = True
                entry_atr = atr[i]
                signal[i] = 1
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [90, 100, 102, 102, 101],
        "high": [91, 102, 103, 103, 102],
        "low": [89, 99, 100.75, 100.25, 100],
        "close": [90, 101.8, 102, 101, 101],
        "volume": [1000] * 5
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 10, 1],
        "ma200": [80] * 5,
        "entry": [True, False, False, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params(breakeven_gain=0.01))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()
    indicators.loc[index[3], "atr"] = 1
    prefix = strategy.generate_signals(data.iloc[:3], indicators.iloc[:3], None, Params(breakeven_gain=0.01))
    assert prefix.data["signal"].equals(result.data["signal"].iloc[:3])


if __name__ == "__main__":
    _self_check()
