from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    history: int = 126
    max_streak: int = 5
    min_samples: int = 8
    edge_threshold: float = 0.12
    atr_period: int = 14
    stop_atr: float = 1.25
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1790938935"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params):
        if (params.history < 1 or params.max_streak < 2
                or params.min_samples < 1 or params.atr_period < 1
                or params.stop_atr <= 0 or params.hold_days not in (1, 2)
                or not 0 < params.edge_threshold < 0.5):
            raise ValueError("Invalid streak strategy parameters")
        return max(params.history + params.max_streak + 1, params.atr_period + 1)

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["direction"] = np.sign(data["close"].diff()).fillna(0)
        alive = out["direction"].ne(0)
        out["streak"] = alive.astype(int)
        for lag in range(1, params.max_streak):
            alive = alive & out["direction"].eq(out["direction"].shift(lag))
            out["streak"] += alive.astype(int)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["continuation"] = out["direction"].eq(out["direction"].shift(1))
        out["probability"] = np.nan
        out["samples"] = 0.0
        for length in range(2, params.max_streak + 1):
            match = (out["streak"].shift(1).eq(length)
                     & out["direction"].ne(0))
            out[f"samples_{length}"] = match.astype(float).rolling(params.history).sum()
            out[f"wins_{length}"] = (match & out["continuation"]).astype(float).rolling(params.history).sum()
            out[f"probability_{length}"] = (
                (out[f"wins_{length}"] + 1.0) / (out[f"samples_{length}"] + 2.0)
            )
            selected = out["streak"].eq(length)
            out.loc[selected, "probability"] = out.loc[selected, f"probability_{length}"]
            out.loc[selected, "samples"] = out.loc[selected, f"samples_{length}"]
        out["edge"] = out["probability"] - 0.5
        out["strength"] = (
            2.0 * out["edge"].abs() * out["samples"] / (out["samples"] + 10.0)
        ).fillna(0.0).clip(0.0, 1.0)
        valid = (out["samples"].ge(params.min_samples)
                 & out["edge"].abs().ge(params.edge_threshold)
                 & out["atr"].gt(0) & np.isfinite(out["atr"]))
        out["candidate"] = (
            out["direction"] * np.sign(out["edge"].fillna(0.0))
        ).where(valid, 0).astype(int)
        out["candidate_size"] = 0.20 + 0.75 * out["strength"]
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.20}, index=data.index)
        side, entry_bar, stop, position_size = 0, -1, np.nan, 0.20
        frozen_atr = np.nan
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    entry = float(data["open"].iloc[i])
                    stop = entry - side * params.stop_atr * frozen_atr
                close = float(data["close"].iloc[i])
                stopped = (close < stop if side == 1 else close > stop)
                expired = i - entry_bar + 1 >= params.hold_days
                if not np.isfinite(stop) or not np.isfinite(close) or stopped or expired:
                    side = 0
            elif i >= warmup - 1:
                candidate = int(indicators["candidate"].iloc[i])
                if candidate:
                    side = candidate
                    entry_bar = i + 1
                    frozen_atr = float(indicators["atr"].iloc[i])
                    position_size = float(indicators["candidate_size"].iloc[i])
            df.iloc[i, 0] = side
            df.iloc[i, 1] = position_size if side else 0.20
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = StreakParams(history=3, max_streak=2, min_samples=1, atr_period=2)
    s = GeneratedStrategy()
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=14),
    )
    features = pd.DataFrame(
        {"candidate": 0, "atr": 1.0, "candidate_size": 0.6}, index=frame.index
    )
    features.loc[frame.index[[5, 8, 11]], "candidate"] = [1, 1, -1]
    frame.loc[frame.index[9], "close"] = 98.0
    frame.loc[frame.index[12], "close"] = 102.0
    result = s.generate_signals(frame, features, None, p).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, -1, 0]
    assert result.loc[frame.index[6], "size"] == 0.6
    assert result["size"].gt(0).all()
    assert s.indicators(frame, p)["candidate"].isin([-1, 0, 1]).all()
