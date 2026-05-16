from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalTideParams:
    rank_window: int = 63
    entry_percentile: float = 0.75
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    max_hold: int = 2
    trend_ma: int = 200
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[SeasonalTideParams]):
    strategy_id = "gen_a1_1778904914"

    @classmethod
    def params_type(cls) -> type[SeasonalTideParams]:
        return SeasonalTideParams

    @staticmethod
    def warmup_bars(params: SeasonalTideParams) -> int:
        # 252 bars give every trading-day-of-month position ~a year of history
        # before the tide table is considered valid.
        return int(max(params.trend_ma, params.rank_window, params.atr_period) + 252)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalTideParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        out = pd.DataFrame(index=data.index)

        # --- seasonal tide table -------------------------------------------
        # trading-day-of-month position (0-based) within each calendar month
        month = pd.Series(data.index.to_period("M"), index=data.index)
        tdom = month.groupby(month).cumcount()

        r = close.pct_change()
        # Expanding mean of 1-day returns for each calendar position, using
        # PRIOR months only (shift(1) inside each group) -> no lookahead.
        # This is the learned "tide level" for that position.
        tide = r.groupby(tdom).transform(lambda s: s.shift(1).expanding().mean())
        tide = tide.fillna(0.0)

        # Rolling percentile rank of the current tide level: the entry gate is
        # a percentile threshold, not a fixed return level.
        rank_pct = tide.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        # --- ATR for the fixed volatility stop -----------------------------
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        trend = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()

        out["tide"] = tide
        out["rank_pct"] = rank_pct
        out["atr"] = atr
        out["trend_ma"] = trend
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalTideParams,
    ) -> SignalFrame:
        close = data["close"]
        rank_pct = indicators["rank_pct"]
        atr = indicators["atr"]
        trend = indicators["trend_ma"]

        # Entry candidate: current calendar position is a high-percentile
        # "high tide". NaN comparisons evaluate to False (NaN-safe).
        entry_ok = rank_pct >= params.entry_percentile
        if params.use_trend_filter:
            entry_ok = entry_ok & (close > trend)
        entry_ok = entry_ok.fillna(False).to_numpy(dtype=bool)

        close_np = close.to_numpy(dtype=float)
        atr_np = atr.ffill().fillna(0.0).to_numpy(dtype=float)

        n = len(data)
        sig = np.zeros(n, dtype=int)
        pos = 0
        stop_level = 0.0
        hold = 0
        k = float(params.atr_stop_mult)
        max_hold = int(params.max_hold)

        # Path-dependent state machine: enter on the seasonal gate, exit on a
        # FIXED volatility stop (entry close - k*ATR, set once at entry) or
        # after max_hold bars.
        for i in range(n):
            if pos == 0:
                if entry_ok[i] and atr_np[i] > 0.0:
                    pos = 1
                    hold = 0
                    stop_level = close_np[i] - k * atr_np[i]
                    sig[i] = 1
            else:
                hold += 1
                if close_np[i] <= stop_level:
                    pos = 0
                    sig[i] = 0
                elif hold >= max_hold:
                    pos = 0
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
