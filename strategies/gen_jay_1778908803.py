from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 60
    compress_pct: int = 25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778908803'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.lookback * 2 + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        rng = (data['high'] - data['low']) / data['close'].replace(0.0, np.nan)

        window = params.lookback
        roll_min = rng.rolling(window, min_periods=window).min()
        roll_max = rng.rolling(window, min_periods=window).max()
        span = (roll_max - roll_min).replace(0.0, np.nan)
        rng_rank = ((rng - roll_min) / span).clip(0.0, 1.0)
        ind['rng_rank'] = rng_rank

        threshold = params.compress_pct / 100.0
        compressed = (rng_rank < threshold).astype(float)
        ind['compressed'] = compressed

        half = max(params.lookback // 2, 5)
        recent_inf = compressed.rolling(half, min_periods=half).sum()
        prior_inf = compressed.shift(half).rolling(half, min_periods=half).sum()
        ind['R_number'] = recent_inf / (prior_inf + 1.0)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        compressed_arr = indicators['compressed'].values
        R_number_arr = indicators['R_number'].values
        n = len(data)

        signal = np.zeros(n, dtype=int)
        in_trade = False

        for i in range(n):
            c = compressed_arr[i]
            r = R_number_arr[i]

            if np.isnan(c) or np.isnan(r):
                signal[i] = 0
                in_trade = False
                continue

            if in_trade:
                if c == 0.0:
                    in_trade = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if c == 1.0 and r < 1.0:
                    in_trade = True
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column='signal', size_column='size')
