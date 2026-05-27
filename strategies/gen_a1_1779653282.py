from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyStreakParams:
    vol_window: int = 10
    vol_baseline: int = 60
    trend_window: int = 20
    streak_threshold: int = 4
    target_annual_vol: float = 0.15
    base_size: float = 1.0
    max_leverage: float = 1.0
    min_size: float = 0.05


def _streak(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(int)
    grp = (c == 0).cumsum()
    return c.groupby(grp).cumsum().astype(float)


class GeneratedStrategy(BaseStrategy[PredatorPreyStreakParams]):
    strategy_id = "gen_a1_1779653282"

    @classmethod
    def params_type(cls):
        return PredatorPreyStreakParams

    def warmup_bars(self, params: PredatorPreyStreakParams) -> int:
        return int(max(params.vol_baseline, params.trend_window) + params.vol_window + params.streak_threshold + 5)

    def indicators(self, data: pd.DataFrame, params: PredatorPreyStreakParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        short_vol = ret.rolling(params.vol_window, min_periods=params.vol_window).std()
        baseline = short_vol.rolling(params.vol_baseline, min_periods=params.vol_baseline).mean()
        sma = close.rolling(params.trend_window, min_periods=params.trend_window).mean()

        valid_vol = short_vol.notna() & baseline.notna()
        vol_below = (short_vol < baseline) & valid_vol
        vol_above = (short_vol > baseline) & valid_vol

        valid_trend = sma.notna()
        prey_up = (close > sma) & valid_trend
        prey_down = (close < sma) & valid_trend

        vol_dormant_streak = _streak(vol_below)
        vol_active_streak = _streak(vol_above)
        prey_up_streak = _streak(prey_up)
        prey_down_streak = _streak(prey_down)

        ann_vol = short_vol * np.sqrt(252.0)

        ind = pd.DataFrame(
            {
                "short_vol": short_vol,
                "baseline_vol": baseline,
                "sma": sma,
                "vol_dormant_streak": vol_dormant_streak,
                "vol_active_streak": vol_active_streak,
                "prey_up_streak": prey_up_streak,
                "prey_down_streak": prey_down_streak,
                "ann_vol": ann_vol,
            },
            index=data.index,
        )
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorPreyStreakParams,
    ) -> SignalFrame:
        n = int(params.streak_threshold)

        long_entry = (
            (indicators["vol_dormant_streak"] >= n)
            & (indicators["prey_up_streak"] >= n)
        ).fillna(False).to_numpy()
        short_entry = (
            (indicators["vol_active_streak"] >= n)
            & (indicators["prey_down_streak"] >= n)
        ).fillna(False).to_numpy()

        nbars = len(data.index)
        sig_arr = np.zeros(nbars, dtype=np.int64)
        pos = 0
        for i in range(nbars):
            if long_entry[i]:
                pos = 1
            elif short_entry[i]:
                pos = -1
            sig_arr[i] = pos

        signal_raw = pd.Series(sig_arr, index=data.index, dtype=int)

        ann_vol = indicators["ann_vol"].replace(0.0, np.nan)
        raw_size = (params.target_annual_vol / ann_vol) * params.base_size
        size = raw_size.clip(lower=params.min_size, upper=params.max_leverage)
        size = size.fillna(params.min_size).astype(float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal_raw.shift(1).fillna(0).astype(int)
        df["size"] = size.values

        return SignalFrame(data=df, signal_column="signal", size_column="size")
