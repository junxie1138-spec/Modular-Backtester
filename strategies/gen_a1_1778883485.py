from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonTideParams:
    roc_period: int = 5
    accel_period: int = 3
    season_window: int = 12
    season_min: int = 6
    season_threshold: float = 0.0
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_k: float = 2.5
    init_stop_mult: float = 2.0
    max_hold: int = 6


class GeneratedStrategy(BaseStrategy[SeasonTideParams]):
    strategy_id = "gen_a1_1778883485"

    @classmethod
    def params_type(cls):
        return SeasonTideParams

    @staticmethod
    def warmup_bars(params: SeasonTideParams) -> int:
        season_lb = (params.season_window + 1) * 5
        roc_lb = params.roc_period + params.accel_period + 1
        return int(max(season_lb, roc_lb, params.atr_period + 1) + 5)

    def indicators(self, data: pd.DataFrame, params: SeasonTideParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Primitive 2 source: rate-of-change and its acceleration (change in ROC).
        roc = close.pct_change(params.roc_period)
        accel = roc.diff(params.accel_period)

        # Primitive 1 source: per-weekday trailing mean return ('tide table').
        # For each weekday, take only same-weekday bars, shift by one occurrence
        # so the current bar is excluded, then a rolling mean of past tides.
        ret = close.pct_change()
        dow = pd.Series(data.index.dayofweek, index=data.index)
        season = pd.Series(np.nan, index=data.index, dtype=float)
        for d in range(5):
            mask = (dow == d)
            sub = ret[mask]
            if len(sub) == 0:
                continue
            tide = sub.shift(1).rolling(
                params.season_window, min_periods=params.season_min
            ).mean()
            season.loc[mask] = tide

        # ATR for the breakeven-then-trail exit.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["accel"] = accel
        out["season"] = season
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonTideParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        season = indicators["season"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        bars_held = 0
        breakeven_done = False

        for i in range(n):
            if not in_pos:
                # Two-primitive AND: favorable weekday tide AND accelerating ROC.
                entry_ok = (
                    np.isfinite(season[i])
                    and np.isfinite(accel[i])
                    and season[i] > params.season_threshold
                    and accel[i] > 0.0
                )
                if entry_ok:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    breakeven_done = False
                    a = atr[i] if np.isfinite(atr[i]) else entry_price * 0.02
                    stop = entry_price - params.init_stop_mult * a
                    pos[i] = 1
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                a = atr[i] if np.isfinite(atr[i]) else entry_price * 0.02

                # Breakeven: once +breakeven_pct is reached, lift stop to entry.
                if (not breakeven_done) and close[i] >= entry_price * (
                    1.0 + params.breakeven_pct
                ):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_done = True

                # Trail: after breakeven, ratchet stop up by k*ATR, never down.
                if breakeven_done:
                    trail = close[i] - params.trail_k * a
                    if trail > stop:
                        stop = trail

                exit_now = (low[i] <= stop) or (bars_held >= params.max_hold)
                if exit_now:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
