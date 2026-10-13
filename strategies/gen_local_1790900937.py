from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    atr_window: int = 12
    expansion_penalty: float = 1.0
    strength_threshold: float = 0.2

    def __post_init__(self):
        if self.atr_window < 2:
            raise ValueError('atr_window must be at least 2')
        if not np.isfinite(self.expansion_penalty) or self.expansion_penalty < 0:
            raise ValueError('expansion_penalty must be finite and nonnegative')
        if not np.isfinite(self.strength_threshold) or self.strength_threshold <= 0:
            raise ValueError('strength_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = 'gen_local_1790900937'

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(params.atr_window + 2)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['intraday_range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['intraday_range', 'high_gap', 'low_gap']].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['prior_atr'] = out['atr'].shift(1).where(lambda s: s > 0)
        out['advancement'] = data['close'].diff()
        out['range_expansion'] = out['true_range'].diff().clip(lower=0)
        out['strength'] = (
            out['advancement'] - params.expansion_penalty * out['range_expansion']
        ) / out['prior_atr']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        strength = indicators['strength']
        eligible = np.isfinite(strength) & (strength > params.strength_threshold)
        df = pd.DataFrame(index=data.index)
        # The same predicate opens and maintains a long; its complement exits.
        df['signal'] = eligible.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = ShockwaveParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'strength': [np.nan, 0.3, 0.4, 0.2, 0.5, -0.2]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= params.atr_window + 2
