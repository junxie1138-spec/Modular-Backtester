from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entry_z: float = 1.2
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791327607"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 51

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or not 0 < params.entry_z < 2.5:
            raise ValueError("entry_z must be between zero and 2.5")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["ma"] = data["close"].rolling(40, min_periods=40).mean()
        out["std"] = data["close"].rolling(40, min_periods=40).std(ddof=0)
        out["z"] = ((data["close"] - out["ma"]) / out["std"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(14, min_periods=14).mean()
        out["positive_spike"] = (out["z"] >= 2.5).astype(int)
        out["negative_spike"] = (out["z"] <= -2.5).astype(int)
        out["positive_memory"] = out["positive_spike"].shift(1).rolling(10, min_periods=10).max()
        out["negative_memory"] = out["negative_spike"].shift(1).rolling(10, min_periods=10).max()
        out["recent_spike"] = (out["positive_spike"] + out["negative_spike"]).shift(1).rolling(3, min_periods=3).max()
        ready = out[["z", "previous_z", "atr", "positive_memory", "negative_memory", "recent_spike"]].notna().all(axis=1)
        ready &= (out["atr"] > 0) & (out["recent_spike"] == 0)
        long_entry = ready & (out["positive_memory"] == 1) & (out["negative_memory"] == 0)
        long_entry &= (out["previous_z"] <= params.entry_z) & (out["z"] > params.entry_z) & (out["z"] < 2.5)
        short_entry = ready & (out["negative_memory"] == 1) & (out["positive_memory"] == 0)
        short_entry &= (out["previous_z"] >= -params.entry_z) & (out["z"] < -params.entry_z) & (out["z"] > -2.5)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if position:
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                    else:
                        entry_bar = i
                        # Anchor to the next-bar open; freeze the decision-bar ATR.
                        stop = opens[i] - position * params.stop_atr * entry_atr
                if position:
                    stopped = (position == 1 and closes[i] < stop) or (position == -1 and closes[i] > stop)
                    if not np.isfinite(closes[i]) or stopped or i - entry_bar + 1 >= 18:
                        position = 0
                        entry_bar = -1
                # An exit always produces a flat decision before another entry.
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_atr = atr[i]
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": [100, 100, 110, 112, 104, 90, 90, 96, 96],
        "close": [100, 100, 120, 112, 104, 90, 80, 96, 96],
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 2.0,
        "candidate": [0, 1, 0, 0, 0, -1, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    # Verify one-bar delay, next-open anchoring, fixed stops, and both directions.
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=23)
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "candidate": 0}, index=index)
    indicators.iloc[1, indicators.columns.get_loc("candidate")] = 1
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20:].eq(0).all()


if __name__ == "__main__":
    _self_check()
