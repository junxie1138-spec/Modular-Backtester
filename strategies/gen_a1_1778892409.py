from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendStreakParams:
    ma_len: int = 50
    pct_len: int = 120
    atr_len: int = 14
    entry_pct: float = 0.85
    exit_pct: float = 0.50
    k_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[TrendStreakParams]):
    """Trend-strength via days-above-MA persistence streak.

    Counts the consecutive run of bars whose close holds above a moving
    average. Enters long when that streak length crosses a rolling
    percentile high of its own recent history (regime-relative trend
    strength). A hysteresis band blocks re-entry until the streak rank
    collapses below a lower percentile. Exit is a fixed ATR volatility
    stop set at entry, with a trend-break backstop.
    """

    strategy_id = "gen_a1_1778892409"

    @classmethod
    def params_type(cls) -> type[TrendStreakParams]:
        return TrendStreakParams

    @staticmethod
    def warmup_bars(params: TrendStreakParams) -> int:
        return int(max(params.ma_len, params.atr_len) + params.pct_len + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendStreakParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()

        # consecutive streak of bars with close above the moving average
        is_above = (close > ma) & ma.notna()
        block = (~is_above).cumsum()
        streak = is_above.astype(int).groupby(block).cumsum()
        streak = streak.astype(float).where(ma.notna(), other=np.nan)

        # streak length expressed as its own rolling percentile rank
        streak_rank = streak.rolling(
            params.pct_len, min_periods=params.pct_len
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["streak"] = streak
        out["streak_rank"] = streak_rank
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendStreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        streak_rank = indicators["streak_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        sig = np.zeros(n, dtype=int)

        in_pos = False
        armed = True
        stop_level = 0.0

        entry_pct = float(params.entry_pct)
        exit_pct = float(params.exit_pct)
        k_atr = float(params.k_atr)

        for i in range(n):
            sr = streak_rank[i]
            a = atr[i]
            stk = streak[i]
            valid = not (np.isnan(sr) or np.isnan(a) or np.isnan(stk))

            # hysteresis: re-arm only after the streak rank collapses low
            if valid and sr < exit_pct:
                armed = True

            if in_pos:
                exit_now = False
                if close[i] <= stop_level:
                    exit_now = True
                elif valid and stk <= 0.0:
                    exit_now = True
                elif valid and sr < exit_pct:
                    exit_now = True
                if exit_now:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                if armed and valid and sr >= entry_pct:
                    in_pos = True
                    armed = False
                    stop_level = close[i] - k_atr * a
                    sig[i] = 1
                else:
                    sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
