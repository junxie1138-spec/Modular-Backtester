from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_min: int = 3
    intensity_window: int = 60
    trend_entry_pct: float = 0.70
    trend_exit_pct: float = 0.45
    rev_entry_pct: float = 0.30
    rev_exit_pct: float = 0.55
    atr_period: int = 14
    atr_mult: float = 2.0
    size_streak_cap: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778914784'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.intensity_window, params.atr_period) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data['close']
        high = data['high']
        low = data['low']

        ret = close.pct_change()
        sign_raw = np.sign(ret)
        sign_filled = sign_raw.fillna(0)

        streak_reset = (sign_filled != sign_filled.shift(1).fillna(0)) | (sign_filled == 0)
        group_id = streak_reset.cumsum()
        raw_count = group_id.groupby(group_id).cumcount() + 1
        streak_count = raw_count.where(sign_filled != 0, 0).astype(float)

        ind['streak_count'] = streak_count
        ind['streak_dir'] = sign_filled
        ind['streak_intensity'] = streak_count * ret.abs()
        ind['intensity_pct'] = (
            ind['streak_intensity']
            .rolling(params.intensity_window, min_periods=params.intensity_window // 2)
            .rank(pct=True)
        )

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        ind['atr'] = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data['close'].values
        sc_arr = indicators['streak_count'].values
        sd_arr = indicators['streak_dir'].values
        ip_arr = indicators['intensity_pct'].values
        atr_arr = indicators['atr'].values

        signal_arr = np.zeros(n, dtype=int)
        size_arr = np.full(n, 0.5)

        regime = 0
        position = 0
        hwm = 0.0
        lwm = np.inf

        for i in range(1, n):
            ip = ip_arr[i]
            cur_atr = atr_arr[i]

            if np.isnan(ip) or np.isnan(cur_atr):
                signal_arr[i] = 0
                size_arr[i] = 0.5
                continue

            if regime == 0:
                if ip >= params.trend_entry_pct:
                    regime = 1
                elif ip <= params.rev_entry_pct:
                    regime = -1
            elif regime == 1:
                if ip < params.trend_exit_pct:
                    regime = 0
            else:
                if ip > params.rev_exit_pct:
                    regime = 0

            if position == 1:
                hwm = max(hwm, close[i])
                if close[i] < hwm - params.atr_mult * cur_atr:
                    position = 0
                    signal_arr[i] = 0
                    size_arr[i] = 0.5
                    continue
            elif position == -1:
                lwm = min(lwm, close[i])
                if close[i] > lwm + params.atr_mult * cur_atr:
                    position = 0
                    signal_arr[i] = 0
                    size_arr[i] = 0.5
                    continue

            cur_sc = sc_arr[i]
            cur_sd = int(sd_arr[i])
            raw_sig = 0

            if cur_sc >= params.streak_min and regime != 0:
                raw_sig = cur_sd if regime == 1 else -cur_sd

            if raw_sig != 0 and raw_sig != position:
                position = raw_sig
                if position == 1:
                    hwm = close[i]
                    lwm = np.inf
                else:
                    lwm = close[i]
                    hwm = 0.0

            signal_arr[i] = position
            if position != 0:
                size_arr[i] = min(float(cur_sc) / float(params.size_streak_cap), 1.0)
            else:
                size_arr[i] = 0.5

        df = pd.DataFrame({'signal': signal_arr, 'size': size_arr}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.5)
        df['size'] = df['size'].clip(lower=0.01)

        return SignalFrame(data=df, signal_column='signal', size_column='size')
