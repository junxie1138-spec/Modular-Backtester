from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    streak_bars: int = 2
    compression_ratio: float = 0.85
    max_body_fraction: float = 0.45
    profit_target: float = 0.006
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791263148'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        if not 1 <= params.streak_bars <= 9:
            raise ValueError('streak_bars must be between 1 and 9')
        if not 1 <= params.holding_bars <= 2:
            raise ValueError('holding_bars must be 1 or 2')
        if not 0 < params.compression_ratio < 1:
            raise ValueError('compression_ratio must be between 0 and 1')
        if not 0 < params.max_body_fraction <= 1:
            raise ValueError('max_body_fraction must be between 0 and 1')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        return int(params.streak_bars + 1)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['valid'] = (
            np.isfinite(data[['open', 'high', 'low', 'close', 'volume']]).all(axis=1)
            & (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            & (data['volume'] > 0)
            & (data['high'] > data['low'])
            & (data['high'] >= data[['open', 'close']].max(axis=1))
            & (data['low'] <= data[['open', 'close']].min(axis=1))
        ).fillna(False)
        out['price_range'] = (data['high'] - data['low']).where(out['valid'])
        out['relative_range'] = out['price_range'] / data['close'].where(out['valid'])
        out['close_change'] = data['close'].where(out['valid']).diff()
        out['body_fraction'] = (data['close'] - data['open']).abs() / out['price_range']
        out['contraction'] = (
            out['valid']
            & out['valid'].shift(1, fill_value=False)
            & (out['relative_range'] < out['relative_range'].shift(1) * params.compression_ratio)
            & (data['volume'] > data['volume'].shift(1))
            & (out['close_change'] < 0)
        ).fillna(False)
        alive = out['contraction'].copy()
        streak = alive.astype(int)
        for lag in range(1, params.streak_bars):
            alive = alive & out['contraction'].shift(lag, fill_value=False)
            streak = streak + alive.astype(int)
        out['streak'] = streak
        out['entry'] = (
            out['valid']
            & (out['streak'] >= params.streak_bars)
            & (data['close'] < data['open'])
            & (out['body_fraction'] <= params.max_body_fraction)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            # Yesterday's decision enters at today's open; exits fill next open.
            if i > 0 and desired[i - 1] == 1 and not np.isfinite(entry_price):
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                entry_price = opens[i]
                age = 0
            if np.isfinite(entry_price):
                age += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or age >= params.holding_bars:
                    entry_price = np.nan
                    desired[i] = 0
                else:
                    desired[i] = 1
            elif entries[i]:
                desired[i] = 1
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = CompressionParams()
    sample = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [103.0, 102.0, 101.5, 102.0, 102.0, 102.0],
        'low': [97.0, 98.0, 98.5, 98.0, 98.0, 98.0],
        'close': [100.0, 99.8, 99.6, 100.8, 100.0, 100.0],
        'volume': [100.0, 200.0, 300.0, 100.0, 100.0, 100.0],
    }, index=pd.date_range('2020-01-01', periods=6))
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 0, 0]
    sample.loc[sample.index[3], 'close'] = 100.0
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 1, 0]
    assert strategy.warmup_bars(params) <= 10
