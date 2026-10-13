from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    history: int = 180
    neighbors: int = 8
    edge: float = 0.004
    atr_period: int = 14
    breakeven_pct: float = 0.04
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713414"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.history + params.max_hold + 3, params.atr_period + 2)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        gap = (data["open"] - prev_close) / atr.replace(0, np.nan)
        return pd.DataFrame({"atr": atr, "gap": gap}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        raw = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        width, horizon = 3, params.max_hold
        warm = cls.warmup_bars(params)

        for t in range(warm, n):
            shape = gap[t - width + 1:t + 1]
            if not np.isfinite(shape).all():
                continue
            first = max(width - 1, t - params.history)
            last = t - horizon - 1
            if last < first:
                continue
            starts = np.arange(first, last + 1)
            candidates = np.column_stack([gap[starts - 2], gap[starts - 1], gap[starts]])
            valid = np.isfinite(candidates).all(axis=1) & np.isfinite(close[starts]) & np.isfinite(close[starts + horizon])
            candidates, starts = candidates[valid], starts[valid]
            if not len(starts):
                continue
            distances = np.mean((candidates - shape) ** 2, axis=1)
            k = min(max(1, params.neighbors), len(starts))
            chosen = np.argpartition(distances, k - 1)[:k]
            outcomes = close[starts[chosen] + horizon] / close[starts[chosen]] - 1.0
            forecast = float(np.mean(outcomes))
            if forecast > params.edge:
                raw[t] = 1
            elif forecast < -params.edge:
                raw[t] = -1
            else:
                continue
            sizes[t] = float(np.clip(abs(forecast) / max(params.edge * 4.0, 1e-9), 0.1, 1.0))

        # Simulate the required breakeven-then-trail exit against subsequent OHLC bars.
        for i in range(n):
            if raw[i] == 0:
                continue
            side, entry = raw[i], close[i]
            entry_atr = atr[i]
            if not np.isfinite(entry_atr) or entry_atr <= 0:
                raw[i] = 0
                continue
            stop = entry - side * params.trail_atr * entry_atr
            best = entry
            for j in range(i + 1, min(n, i + horizon + 1)):
                best = max(best, high[j]) if side == 1 else min(best, low[j])
                if side == 1 and best >= entry * (1.0 + params.breakeven_pct):
                    stop = max(stop, entry, best - params.trail_atr * atr[j - 1] if np.isfinite(atr[j - 1]) else stop)
                elif side == -1 and best <= entry * (1.0 - params.breakeven_pct):
                    stop = min(stop, entry, best + params.trail_atr * atr[j - 1] if np.isfinite(atr[j - 1]) else stop)
                stopped = low[j] <= stop if side == 1 else high[j] >= stop
                timed_out = j == min(n, i + horizon + 1) - 1
                if stopped or timed_out:
                    raw[i] = 0
                    break

        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
