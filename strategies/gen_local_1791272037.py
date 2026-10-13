from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    lookback: int = 20
    compression_ratio: float = 0.7

    def __post_init__(self):
        if self.lookback < 4:
            raise ValueError('lookback must be at least 4')
        if not 0.0 < self.compression_ratio <= 1.0:
            raise ValueError('compression_ratio must be in (0, 1]')


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = 'gen_local_1791272037'

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        return max(20, params.lookback + 15)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        # ponytail: fixed 16-tap fractional filter; extend taps if truncation materially changes results.
        weights = np.ones(16)
        for k in range(1, 16):
            weights[k] = -weights[k - 1] * (0.45 - k + 1) / k
        fractional_close = sum(weights[k] * close.shift(k) for k in range(16))
        channel_width = high.rolling(4).max() - low.rolling(4).min()
        typical_width = channel_width.shift(4).rolling(params.lookback).mean()
        return pd.DataFrame({
            'fractional_close': fractional_close,
            'recoil': fractional_close.diff(4),
            'equilibrium': fractional_close.rolling(params.lookback).mean(),
            'channel_width': channel_width,
            'typical_width': typical_width,
            'compression': channel_width / typical_width.where(typical_width > 0),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators).all(axis=1)
        entry_condition = (
            valid
            & indicators['channel_width'].ge(0)
            & indicators['compression'].lt(params.compression_ratio)
            & indicators['recoil'].gt(0)
            & indicators['fractional_close'].lt(indicators['equilibrium'])
        )
        df = pd.DataFrame(index=data.index)
        # A true-to-false qualification flip closes the position; no timed exit.
        df['signal'] = entry_condition.fillna(False).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoilParams()
    index = pd.date_range('2020-01-01', periods=96)
    close = 100.0 + np.arange(96) * 0.03 + np.sin(np.arange(96) / 3)
    data = pd.DataFrame({
        'open': close,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': 1000000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert np.isfinite(derived.iloc[strategy.warmup_bars(params) - 1:]).all().all()
    pd.testing.assert_frame_equal(
        derived.iloc[:64], strategy.indicators(data.iloc[:64], params)
    )
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals['signal'].isin([0, 1]).all()
    assert signals['size'].gt(0).all()
    assert signals['signal'].iloc[0] == 0
    synthetic = pd.DataFrame({
        'fractional_close': [0.0] * 5,
        'recoil': [-1.0, 1.0, 1.0, -1.0, 1.0],
        'equilibrium': [1.0] * 5,
        'channel_width': [0.5] * 5,
        'typical_width': [1.0] * 5,
        'compression': [0.5] * 5,
    }, index=index[:5])
    result = strategy.generate_signals(data.iloc[:5], synthetic, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
