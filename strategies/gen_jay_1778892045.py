from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStreakFadeParams:
    streak_threshold: int = 3
    confirm_bars: int = 2
    profit_target_pct: float = 2.0
    max_hold_bars: int = 10
    min_gap_pct: float = 0.15


class GeneratedStrategy(BaseStrategy[GapStreakFadeParams]):
    strategy_id = "gen_jay_1778892045"

    @classmethod
    def params_type(cls):
        return GapStreakFadeParams

    @staticmethod
    def warmup_bars(params: GapStreakFadeParams) -> int:
        return params.streak_threshold + params.confirm_bars + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapStreakFadeParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        gap_pct = (data["open"] - prev_close) / prev_close * 100.0

        gap_up = (gap_pct >= params.min_gap_pct).astype(int)
        gap_down = (gap_pct <= -params.min_gap_pct).astype(int)

        # Vectorised consecutive streak counting via cumsum-group trick
        grp_up = (gap_up != gap_up.shift(1)).cumsum()
        ind["streak_up"] = gap_up.groupby(grp_up).cumsum()

        grp_dn = (gap_down != gap_down.shift(1)).cumsum()
        ind["streak_down"] = gap_down.groupby(grp_dn).cumsum()

        body_bull = (data["close"] > data["open"]).astype(int)
        body_bear = (data["close"] < data["open"]).astype(int)

        ind["bull_confirm"] = body_bull.rolling(
            params.confirm_bars, min_periods=params.confirm_bars
        ).sum()
        ind["bear_confirm"] = body_bear.rolling(
            params.confirm_bars, min_periods=params.confirm_bars
        ).sum()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapStreakFadeParams,
    ) -> SignalFrame:
        n = len(data)
        closes = data["close"].values
        profit_target = params.profit_target_pct / 100.0

        streak_up = indicators["streak_up"].fillna(0).values
        streak_down = indicators["streak_down"].fillna(0).values
        bull_confirm = indicators["bull_confirm"].fillna(0).values
        bear_confirm = indicators["bear_confirm"].fillna(0).values

        # Long: N consecutive gap-downs, last confirm_bars candles all bullish bodies
        # Short: N consecutive gap-ups,  last confirm_bars candles all bearish bodies
        raw_long = (streak_down >= params.streak_threshold) & (
            bull_confirm == params.confirm_bars
        )
        raw_short = (streak_up >= params.streak_threshold) & (
            bear_confirm == params.confirm_bars
        )

        sig = np.zeros(n, dtype=np.int32)
        in_position = False
        direction = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_position:
                bars_held += 1
                pnl = (
                    (closes[i] - entry_price) / entry_price
                    if direction == 1
                    else (entry_price - closes[i]) / entry_price
                )
                if pnl >= profit_target or bars_held >= params.max_hold_bars:
                    sig[i] = 0
                    in_position = False
                    direction = 0
                    bars_held = 0
                else:
                    sig[i] = direction
            else:
                if raw_long[i]:
                    sig[i] = 1
                    in_position = True
                    direction = 1
                    entry_price = closes[i]
                    bars_held = 0
                elif raw_short[i]:
                    sig[i] = -1
                    in_position = True
                    direction = -1
                    entry_price = closes[i]
                    bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
