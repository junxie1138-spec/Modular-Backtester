from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    tail: float = 0.12
    quiet_tail: float = 0.35
    atr_period: int = 14
    stop_atr: float = 1.6
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791256897"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.atr_period + params.rank_window + 4))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.rank_window >= 2 and params.atr_period >= 1
                and 0 < params.tail < 0.5 and 0 < params.quiet_tail < 1
                and params.stop_atr > 0 and 3 <= params.hold_days <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["ma200"] = close.rolling(200).mean()
        out["daily_change"] = close.diff()
        out["displacement"] = close.diff(3) / out["atr"].replace(0, np.nan)
        out["participation"] = data["volume"].rolling(3).sum()
        out["stretch_rank"] = out["displacement"].rolling(params.rank_window).rank(pct=True)
        out["participation_rank"] = out["participation"].rolling(params.rank_window).rank(pct=True)
        out["prior_stretch_rank"] = out["stretch_rank"].shift(1)
        out["prior_participation_rank"] = out["participation_rank"].shift(1)
        out["prior_displacement"] = out["displacement"].shift(1)
        required = ["atr", "ma200", "daily_change", "prior_stretch_rank",
                    "prior_participation_rank", "prior_displacement"]
        out["ready"] = np.isfinite(out[required]).all(axis=1) & out["atr"].gt(0)
        quiet = out["prior_participation_rank"].le(params.quiet_tail)
        out["long_entry"] = (
            out["ready"] & quiet & close.gt(out["ma200"])
            & out["prior_stretch_rank"].le(params.tail)
            & out["prior_displacement"].lt(0) & out["daily_change"].gt(0)
        ).fillna(False)
        out["short_entry"] = (
            out["ready"] & quiet & close.lt(out["ma200"])
            & out["prior_stretch_rank"].ge(1 - params.tail)
            & out["prior_displacement"].gt(0) & out["daily_change"].lt(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry_atr = high_water = low_water = 0.0
        high_stop = -np.inf
        short_stop = np.inf
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if not ready[i] or not np.isfinite(close[i]) or not np.isfinite(opening[i]):
                position = 0
                age = 0
                continue
            if position:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    high_water = low_water = opening[i]
                    high_stop = high_water - params.stop_atr * entry_atr
                    short_stop = low_water + params.stop_atr * entry_atr
                age += 1
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                high_stop = max(high_stop, high_water - params.stop_atr * entry_atr)
                short_stop = min(short_stop, low_water + params.stop_atr * entry_atr)
                # Required high-water exit also closes shorts after a downside release.
                stopped = close[i] <= high_stop
                if position == -1:
                    stopped = stopped or close[i] >= short_stop
                if stopped or age >= params.hold_days:
                    position = 0
                    age = 0
                    continue
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_atr = atr[i]
                age = 0
            signals[i] = position
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({"open": 100.0, "close": [100., 100., 103., 100., 100., 100.]}, index=index)
    ind = pd.DataFrame({"atr": 2.0, "ready": True,
                        "long_entry": [True, False, False, False, False, False],
                        "short_entry": False}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, Params(stop_atr=1.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    ind["long_entry"] = False
    ind["short_entry"] = [True, False, False, False, False, False]
    data["close"] = [100., 100., 99., 98., 97., 96.]
    result = GeneratedStrategy.generate_signals(data, ind, None, Params(stop_atr=10.0)).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
