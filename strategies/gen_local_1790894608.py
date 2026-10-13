from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    lookback: int = 126
    min_streak: int = 2
    edge_threshold: float = 0.10
    min_observations: int = 8
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    holding_bars: int = 18
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1790894608"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(params.lookback + params.min_streak + 3, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if (params.lookback < 2 or params.min_streak < 1
                or params.min_observations < 1
                or params.min_observations > params.lookback
                or params.atr_window < 1 or params.holding_bars < 1
                or params.cooldown_bars < 0
                or not 0 < params.edge_threshold < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["direction"] = np.sign(data["close"].diff()).fillna(0).astype(int)
        groups = out["direction"].ne(out["direction"].shift()).cumsum()
        out["streak"] = (out["direction"].groupby(groups).cumcount() + 1).where(
            out["direction"].ne(0), 0)
        out["opportunity"] = (
            out["streak"].shift(1).ge(params.min_streak)
            & out["direction"].shift(1).ne(0)).astype(float)
        out["repeat"] = (
            out["direction"].eq(out["direction"].shift(1))
            & out["direction"].ne(0)).astype(float)
        out["eligible"] = (
            out["direction"].shift(1).ne(0)
            & out["direction"].shift(1).notna()).astype(float)
        out["observations"] = out["opportunity"].shift(1).rolling(params.lookback).sum()
        out["continuations"] = (
            out["opportunity"] * out["repeat"]).shift(1).rolling(params.lookback).sum()
        out["baseline"] = (
            out["repeat"].shift(1).rolling(params.lookback).sum()
            / out["eligible"].shift(1).rolling(params.lookback).sum().replace(0, np.nan))
        out["edge"] = (
            out["continuations"] / out["observations"].replace(0, np.nan)
            - out["baseline"])
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = (
            out["streak"].between(params.min_streak, params.min_streak + 4)
            & out["observations"].ge(params.min_observations)
            & out["edge"].abs().ge(params.edge_threshold)
            & out["atr"].gt(0)
            & np.isfinite(out["edge"]) & np.isfinite(out["atr"]))
        out["candidate"] = (out["direction"] * np.sign(out["edge"])).where(
            ready, 0).fillna(0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("confirmed")] = 0
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = np.nan
        entry_atr = np.nan
        armed = False
        age = 0
        next_entry = 0
        for i in range(len(data)):
            atr = float(indicators["atr"].iloc[i])
            close = float(data["close"].iloc[i])
            if position:
                if np.isnan(entry):
                    entry = float(data["open"].iloc[i])
                    stop = position * entry - params.trail_atr * entry_atr
                    age = 0
                age += 1
                adverse = (float(data["low"].iloc[i]) if position == 1
                           else -float(data["high"].iloc[i]))
                # Existing stops are checked before this close can tighten them.
                # Stop exits are next-open decisions under the shifted-signal contract.
                if (not np.isfinite(entry) or entry <= 0
                        or not np.isfinite(close) or not np.isfinite(adverse)
                        or not np.isfinite(atr) or atr <= 0
                        or adverse <= stop or age >= params.holding_bars):
                    position = 0
                    entry = np.nan
                    next_entry = i + params.cooldown_bars + 1
                else:
                    signed_close = position * close
                    if signed_close - position * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed-price stops rise for both directions; raw short stops fall.
                        stop = max(stop, position * entry,
                                   signed_close - params.trail_atr * atr)
            elif i >= next_entry:
                candidate = int(indicators["confirmed"].iloc[i])
                if candidate and np.isfinite(atr) and atr > 0 and np.isfinite(close) and close > 0:
                    position = candidate
                    entry = np.nan
                    entry_atr = atr
                    armed = False
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StreakParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        close = np.array([100, 100, 100, 102, 101, 100, 100], dtype=float)
        close = 100 + side * (close - 100)
        bars = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=index)
        candidate = pd.Series(side, index=index)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        derived = pd.DataFrame({"atr": 1.0, "confirmed": confirmed}, index=index)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(bars, derived, ctx, params).data
        assert result["signal"].tolist() == [0, 0, side, side, side, side, 0]
        assert result["size"].gt(0).all()
        computed = strategy.indicators(bars, params)
        assert computed["confirmed"].eq(0).all()
