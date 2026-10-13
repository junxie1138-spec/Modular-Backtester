from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_drawdown: float = 0.02
    impulse_ratio: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.min_drawdown) or not 0 < self.min_drawdown < 1:
            raise ValueError('min_drawdown must be between zero and one')
        if not np.isfinite(self.impulse_ratio) or self.impulse_ratio <= 0:
            raise ValueError('impulse_ratio must be finite and positive')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791252854'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 11

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['body'] = data['close'] - data['open']
        out['prior_body'] = out['body'].shift(1)
        out['prior_peak'] = data['high'].rolling(10, min_periods=10).max().shift(1)
        out['prior_drawdown'] = 1.0 - data['close'].shift(1) / out['prior_peak'].replace(0, np.nan)
        out['prior_midpoint'] = (data['open'].shift(1) + data['close'].shift(1)) / 2.0
        volume = data['volume'].where(data['volume'] > 0)
        out['buy_impulse'] = out['body'].clip(lower=0) * volume
        out['sell_impulse'] = (-out['prior_body']).clip(lower=0) * volume.shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & data['close'].notna()
        entry = (
            valid
            & (indicators['prior_drawdown'] >= params.min_drawdown)
            & (indicators['prior_body'] < 0)
            & (indicators['body'] > 0)
            & (indicators['sell_impulse'] > 0)
            & (indicators['buy_impulse'] >= params.impulse_ratio * indicators['sell_impulse'])
            & (data['close'] > indicators['prior_midpoint'])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Flat precisely when the entry condition flips false; no timed exit.
        df['signal'] = entry.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    data = pd.DataFrame({
        'open': [100.0] * 15 + [100.0, 97.0, 99.0, 100.0],
        'high': [101.0] * 15 + [100.0, 100.0, 101.0, 101.0],
        'low': [99.0] * 15 + [96.0, 96.0, 98.0, 99.0],
        'close': [100.0] * 15 + [97.0, 99.0, 100.0, 100.0],
        'volume': [100.0] * 16 + [200.0, 100.0, 100.0],
    }, index=pd.date_range('2020-01-01', periods=19))
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] * 17 + [1, 0]
    assert result['signal'].isin([0, 1]).all()
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) == 11
    quiet = data.copy()
    quiet.loc[quiet.index[16], 'volume'] = 100.0
    rejected = strategy.generate_signals(quiet, strategy.indicators(quiet, params), None, params).data
    assert rejected['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
