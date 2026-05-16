from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_short: int = 50
    ma_long: int = 200
    zscore_lookback: int = 60
    zscore_entry: float = -0.3
    decay_span: int = 10
    atr_period: int = 14
    trail_k: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778908573'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_long + params.zscore_lookback + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data['close'].shift(1)
        gap_return = (data['open'] - prev_close) / prev_close

        ind['gap_pressure'] = gap_return.ewm(span=params.decay_span, adjust=False).mean()

        ma_short = data['close'].rolling(params.ma_short).mean()
        dist = data['close'] - ma_short
        dist_std = dist.rolling(params.zscore_lookback).std()
        ind['zscore'] = dist / dist_std.where(dist_std > 0, other=np.nan)

        ind['ma_long'] = data['close'].rolling(params.ma_long).mean()

        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - prev_close).abs(),
            (data['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind['atr'] = tr.ewm(span=params.atr_period, adjust=False).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data['close'].values
        gap_pressure = indicators['gap_pressure'].values
        zscore = indicators['zscore'].values
        ma_long = indicators['ma_long'].values
        atr = indicators['atr'].values
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95, dtype=float)

        in_trade = False
        hwm = 0.0

        for i in range(1, n):
            gp = gap_pressure[i]
            gp_prev = gap_pressure[i - 1]
            zs = zscore[i]
            ml = ma_long[i]
            at = atr[i]
            cl = close[i]

            if np.isnan(gp) or np.isnan(gp_prev) or np.isnan(zs) or np.isnan(ml) or np.isnan(at):
                signal[i] = 0
                if in_trade:
                    in_trade = False
                    hwm = 0.0
                continue

            regime_ok = cl > ml

            if in_trade:
                if cl > hwm:
                    hwm = cl
                stop = hwm - params.trail_k * at
                if cl < stop or not regime_ok:
                    signal[i] = 0
                    in_trade = False
                    hwm = 0.0
                else:
                    signal[i] = 1
            else:
                gap_crossover = gp_prev < 0.0 and gp >= 0.0
                if regime_ok and gap_crossover and zs <= params.zscore_entry:
                    signal[i] = 1
                    in_trade = True
                    hwm = cl
                else:
                    signal[i] = 0

        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column='signal', size_column='size')
