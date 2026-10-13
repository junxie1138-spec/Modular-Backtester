from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTranslationParams:
    lookback: int = 6
    migration_ratio: float = 0.75

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError('lookback must be at least 2')
        if not np.isfinite(self.migration_ratio) or self.migration_ratio <= 0:
            raise ValueError('migration_ratio must be finite and positive')


class GeneratedStrategy(BaseStrategy[RangeTranslationParams]):
    strategy_id = 'gen_local-w2_1791236585'

    @classmethod
    def params_type(cls) -> type[RangeTranslationParams]:
        return RangeTranslationParams

    def warmup_bars(self, params: RangeTranslationParams) -> int:
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RangeTranslationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['high_step'] = high.diff()
        out['low_step'] = low.diff()
        h = out['high_step']
        l = out['low_step']
        valid_step = h.notna() & l.notna()
        out['translation'] = pd.Series(
            np.where(h * l > 0, np.sign(h) * np.minimum(h.abs(), l.abs()), 0.0),
            index=data.index,
        ).where(valid_step)
        out['breathing'] = (h - l).abs()
        out['translation_sum'] = out['translation'].rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out['breathing_sum'] = out['breathing'].rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out['prior_high'] = high.shift(1)
        out['prior_low'] = low.shift(1)
        out['yield_threshold'] = params.migration_ratio * out['breathing_sum']
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = out[['translation_sum', 'breathing_sum', 'prior_high', 'prior_low']].notna().all(axis=1) & close.notna()
        out['bullish'] = valid & (close > out['prior_high']) & (out['translation_sum'] > out['yield_threshold'])
        out['bearish'] = valid & (close < out['prior_low']) & (out['translation_sum'] < -out['yield_threshold'])
        out['setup'] = np.select([out['bullish'], out['bearish']], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTranslationParams,
    ) -> SignalFrame:
        setups = indicators['setup'].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, direction in enumerate(setups):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                # Opposite qualifying entry closes the position for this bar.
                state = 0
            positions[i] = state
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = RangeTranslationParams(lookback=2)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [9.0] * 8,
        'high': [10.0, 11.0, 13.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        'low': [8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        'close': [9.0, 10.0, 12.0, 9.0, 12.0, 13.0, 14.0, 15.0],
        'volume': [1000.0] * 8,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived['translation'].iloc[1:4].tolist() == [1.0, 1.0, -1.0]
    assert derived['setup'].iloc[:2].eq(0).all()
    setups = pd.DataFrame({'setup': [1, 0, -1, 0, -1, 0, 1, 0]}, index=index)
    result = strategy.generate_signals(data, setups, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) == 3


if __name__ == '__main__':
    _self_check()
