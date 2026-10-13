from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 12
    entry_z: float = 1.0
    repair_min: float = 0.25

    def __post_init__(self):
        if self.ma_window < 2:
            raise ValueError('ma_window must be at least 2')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be positive and finite')
        if not np.isfinite(self.repair_min) or not 0 < self.repair_min < 0.65:
            raise ValueError('repair_min must be between 0 and 0.65')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791238477'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.ma_window + 1, 10))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close']
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['dispersion'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['z'] = (close - out['ma']) / out['dispersion'].where(out['dispersion'] > 0)
        out['previous_z'] = out['z'].shift(1)
        deficit = (-out['previous_z']).where(out['previous_z'] < 0)
        out['repair'] = (out['z'] - out['previous_z']) / deficit
        out['peak'] = close.rolling(10, min_periods=10).max()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['price_change'] = close.diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        eligible = (
            (data['close'] > indicators['ma200'])
            & (indicators['previous_z'] <= -params.entry_z)
            & (indicators['z'] < 0)
            & indicators['repair'].between(params.repair_min, 0.65)
            & (indicators['drawdown'] >= 0.015)
            & (indicators['price_change'] > 0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while eligible; exit only when this same entry condition flips false.
        df['signal'] = eligible.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0] * 5}, index=index)
    indicators = pd.DataFrame({
        'ma200': [90.0] * 5,
        'z': [np.nan, -1.0, -0.8, -0.8, -0.6],
        'previous_z': [np.nan, -1.5, -1.2, -1.0, -0.8],
        'repair': [np.nan, 1 / 3, 1 / 3, 0.2, 0.25],
        'drawdown': [0.02] * 5,
        'price_change': [1.0] * 5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RecoveryParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(None, RecoveryParams(ma_window=250)) == 251


if __name__ == '__main__':
    _self_check()
