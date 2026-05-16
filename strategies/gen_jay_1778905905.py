from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 20
    profit_target_pct: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778905905"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # ma_period for zscore rolling, +1 for diff, +ma_period*2 for vel_threshold median
        return params.ma_period * 3 + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]

        ma = close.rolling(params.ma_period).mean()
        std = close.rolling(params.ma_period).std()
        ind["zscore"] = (close - ma) / std.replace(0, np.nan)

        # velocity of z-score: rate at which price moves through the MA node
        ind["zscore_vel"] = ind["zscore"].diff()

        # rolling median of |velocity| across 2x the MA window = resonance threshold
        ind["vel_threshold"] = (
            ind["zscore_vel"].abs().rolling(params.ma_period * 2).median()
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        zscore = indicators["zscore"]
        zscore_vel = indicators["zscore_vel"]
        vel_threshold = indicators["vel_threshold"]

        prev_z = zscore.shift(1)

        # zero-crossing detection: z-score changes sign
        bull_cross = (prev_z < 0) & (zscore >= 0)
        bear_cross = (prev_z > 0) & (zscore <= 0)

        # resonance filter: crossing velocity exceeds historical median in crossing direction
        # bull: velocity positive and large; bear: velocity negative and large in magnitude
        resonant_bull = bull_cross & (zscore_vel > vel_threshold)
        resonant_bear = bear_cross & (zscore_vel < -vel_threshold)

        raw_signal_series = resonant_bull.astype(int) - resonant_bear.astype(int)

        # mask out warmup period where any indicator is NaN
        valid = (
            (~zscore.isna()) & (~zscore_vel.isna()) & (~vel_threshold.isna())
        )
        close_arr = close.values
        raw_arr = np.where(valid.values, raw_signal_series.values, 0).astype(int)
        signal_arr = np.zeros(len(data), dtype=int)

        TIME_STOP = 10  # ~2 trading weeks
        pt = params.profit_target_pct / 100.0

        i = 0
        while i < len(data):
            if raw_arr[i] != 0:
                direction = int(raw_arr[i])
                entry_price = close_arr[i]
                signal_arr[i] = direction
                j = i + 1
                while j < len(data) and (j - i) < TIME_STOP:
                    ret = (
                        (close_arr[j] - entry_price) / entry_price * direction
                        if entry_price > 0
                        else 0.0
                    )
                    if ret >= pt:
                        break
                    signal_arr[j] = direction
                    j += 1
                i = j
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal_arr
        df["size"] = 1.0

        # mandatory 1-bar shift: decision on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
