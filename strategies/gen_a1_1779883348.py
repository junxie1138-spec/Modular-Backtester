from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    gap_streak_threshold: int = 2
    intraday_streak_threshold: int = 2
    profit_target_pct: float = 0.015
    time_stop_bars: int = 4
    min_gap_pct: float = 0.0005
    cooldown_bars: int = 1


def _consecutive_streak(flags: pd.Series) -> pd.Series:
    s = flags.fillna(0).astype(int)
    reset_groups = (s == 0).cumsum()
    return s.groupby(reset_groups).cumsum()


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779883348"

    @classmethod
    def params_type(cls):
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        return int(max(params.gap_streak_threshold, params.intraday_streak_threshold, 2)) + 5

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        gap_ret = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
        intraday_ret = (close - open_) / open_.replace(0.0, np.nan)

        up_gap_flag = (gap_ret > float(params.min_gap_pct)).astype(int)
        intraday_bullish_flag = (intraday_ret > 0.0).astype(int)

        gap_streak = _consecutive_streak(up_gap_flag)
        intraday_streak = _consecutive_streak(intraday_bullish_flag)

        return pd.DataFrame(
            {
                "gap_ret": gap_ret.fillna(0.0),
                "intraday_ret": intraday_ret.fillna(0.0),
                "gap_streak": gap_streak.astype(float),
                "intraday_streak": intraday_streak.astype(float),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap_streak = indicators["gap_streak"].to_numpy(dtype=float)
        intraday_streak = indicators["intraday_streak"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)

        gst = int(params.gap_streak_threshold)
        ist = int(params.intraday_streak_threshold)
        pt = float(params.profit_target_pct)
        ts = int(params.time_stop_bars)
        cd = int(params.cooldown_bars)

        in_pos = False
        entry_px = 0.0
        entry_bar = 0
        cooldown_until = -1

        for i in range(n):
            if in_pos:
                px = close[i]
                bars_held = i - entry_bar
                pnl = (px - entry_px) / entry_px if entry_px > 0.0 else 0.0
                if pnl >= pt or bars_held >= ts:
                    in_pos = False
                    cooldown_until = i + cd
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                if i > cooldown_until:
                    gs = gap_streak[i]
                    is_ = intraday_streak[i]
                    if (
                        not np.isnan(gs)
                        and not np.isnan(is_)
                        and gs >= gst
                        and is_ >= ist
                    ):
                        in_pos = True
                        entry_px = close[i]
                        entry_bar = i
                        raw[i] = 1

        df = pd.DataFrame(
            {
                "signal": raw,
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
