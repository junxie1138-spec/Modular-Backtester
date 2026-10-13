from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapConsensusParams:
    window: int = 6
    reference_bars: int = 126
    percentile: float = 0.85
    hold_bars: int = 18

    def __post_init__(self):
        if self.window < 2 or self.reference_bars < 2 or self.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < self.percentile < 1.0:
            raise ValueError('percentile must be between zero and one.')


class GeneratedStrategy(BaseStrategy[GapConsensusParams]):
    strategy_id = 'gen_local-w2_1791340884'

    @classmethod
    def params_type(cls):
        return GapConsensusParams

    def warmup_bars(self, params):
        return int(2 * params.window + params.reference_bars + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).replace(0.0, np.nan)
        out['return'] = (data['close'] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['gap'] = (data['open'] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['return_direction'] = np.sign(out['return'])
        out['gap_direction'] = np.sign(out['gap'])
        out['recent_breadth'] = out['return_direction'].rolling(params.window, min_periods=params.window).mean()
        out['prior_breadth'] = out['recent_breadth'].shift(params.window)
        out['gap_consensus'] = out['gap_direction'].rolling(params.window, min_periods=params.window).mean()
        out['prior_direction'] = np.sign(out['prior_breadth'])
        out['divergence'] = out['gap_consensus'].abs() * (
            out['prior_breadth'].abs() - out['recent_breadth'] * out['prior_direction']
        )
        out['threshold'] = out['divergence'].shift(1).rolling(
            params.reference_bars, min_periods=params.reference_bars
        ).quantile(params.percentile)
        eligible = (
            (out['gap_consensus'] * out['prior_direction'] > 0.0)
            & (out['recent_breadth'] * out['prior_direction'] < 0.0)
            & (out['divergence'] > out['threshold'])
        )
        out['entry'] = (-out['prior_direction']).where(eligible, 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        direction = 0
        entry_bar = -1
        for bar, candidate in enumerate(entries):
            if direction != 0:
                if bar - entry_bar == params.hold_bars:
                    direction = 0
            elif candidate != 0:
                direction = int(candidate)
                entry_bar = bar
            positions[bar] = direction
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapConsensusParams(hold_bars=18)
    index = pd.date_range('2020-01-01', periods=50, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': [-1] + [1] * 49}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:19] == -1).all()
    assert result['signal'].iloc[19] == 0
    assert (result['signal'].iloc[20:38] == 1).all()
    assert result['signal'].iloc[38] == 0
    assert (result['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
