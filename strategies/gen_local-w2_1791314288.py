from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SupportRankParams:
    lookback: int = 20
    stop_atr: float = 0.8

    def __post_init__(self):
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 3:
            raise ValueError('lookback must be an integer >= 3')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[SupportRankParams]):
    strategy_id = 'gen_local-w2_1791314288'

    @classmethod
    def params_type(cls):
        return SupportRankParams

    def warmup_bars(self, params: SupportRankParams) -> int:
        return int(max(params.lookback, 15))

    def indicators(self, data: pd.DataFrame, params: SupportRankParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['low_rank'] = data['low'].rolling(params.lookback, min_periods=params.lookback).rank(method='average', pct=True)
        out['high_rank'] = data['high'].rolling(params.lookback, min_periods=params.lookback).rank(method='average', pct=True)
        out['rank_lead'] = out['low_rank'] - out['high_rank']
        out['ready'] = np.isfinite(out[['previous_close', 'atr', 'low_rank', 'high_rank']]).all(axis=1) & out['atr'].gt(0)
        out['entry'] = (
            out['ready']
            & out['low_rank'].ge(0.85)
            & out['rank_lead'].ge(0.25)
            & data['close'].gt(data['open'])
            & data['close'].gt(out['previous_close'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SupportRankParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        holding = False
        age = 0
        high_water = np.nan
        stop = np.nan

        for i, price in enumerate(close):
            if not ready[i] or not np.isfinite(price) or price <= 0:
                holding = False
                age = 0
                high_water = stop = np.nan
                continue
            if holding:
                # The first held bar follows the entry decision; only held closes enter the high-water mark.
                age += 1
                high_water = price if not np.isfinite(high_water) else max(high_water, price)
                candidate = high_water - params.stop_atr * atr[i]
                stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                if price <= stop or age >= 2:
                    holding = False
            elif entry[i]:
                holding = True
                age = 0
                high_water = stop = np.nan
            signal[i] = int(holding)

        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    close = pd.Series([100., 101., 102., 100., 101., 102.], index=index)
    data = pd.DataFrame({'open': close, 'high': close + 1., 'low': close - 1., 'close': close, 'volume': 1000.}, index=index)
    indicators = pd.DataFrame({'atr': 1., 'ready': True, 'entry': [True, False, False, False, True, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SupportRankParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1]
    assert result['size'].gt(0).all()
    assert pd.Series([3., 1., 2., 4.]).rolling(3, min_periods=3).rank(pct=True).iloc[-1] == 1.0


if __name__ == '__main__':
    _self_check()
