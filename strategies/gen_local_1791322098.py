from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalExtinctionParams:
    complexity_floor: float = 0.65
    target_daily_vol: float = 0.008
    hold_bars: int = 2

    def __post_init__(self):
        if not 0.0 <= self.complexity_floor <= 5.0 / 6.0:
            raise ValueError('complexity_floor must be between zero and 5/6')
        if not np.isfinite(self.target_daily_vol) or self.target_daily_vol <= 0:
            raise ValueError('target_daily_vol must be finite and positive')
        if isinstance(self.hold_bars, bool) or not isinstance(self.hold_bars, int) or self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')


class GeneratedStrategy(BaseStrategy[OrdinalExtinctionParams]):
    strategy_id = 'gen_local_1791322098'

    @classmethod
    def params_type(cls):
        return OrdinalExtinctionParams

    def warmup_bars(self, params: OrdinalExtinctionParams) -> int:
        # One return lag, two ordinal lags, six motifs, one complexity lag.
        return 10

    def indicators(self, data: pd.DataFrame, params: OrdinalExtinctionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag1'] = out['return'].shift(1)
        out['return_lag2'] = out['return'].shift(2)
        a, b, c = out['return_lag2'], out['return_lag1'], out['return']
        valid = a.notna() & b.notna() & c.notna()
        # Temporal ordering breaks ties deterministically.
        code = (a <= b).astype(int) + 2 * (b <= c).astype(int) + 4 * (a <= c).astype(int)
        out['ordinal_pattern'] = code.where(valid)
        probability_columns = []
        for pattern in (0, 1, 2, 5, 6, 7):
            column = f'probability_{pattern}'
            occurrence = out['ordinal_pattern'].eq(pattern).astype(float).where(valid)
            out[column] = occurrence.rolling(6, min_periods=6).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        out['complexity'] = (1.0 - probabilities.pow(2).sum(axis=1, min_count=6)).clip(0.0, 5.0 / 6.0)
        out['previous_complexity'] = out['complexity'].shift(1)
        out['daily_vol'] = out['return'].rolling(6, min_periods=6).std(ddof=0)
        extinction = (out['previous_complexity'] >= params.complexity_floor) & (out['complexity'] < out['previous_complexity'] - 1e-12)
        accelerating_up = (out['ordinal_pattern'] == 7) & (a > 0) & (b > a) & (c > b)
        accelerating_down = (out['ordinal_pattern'] == 0) & (a < 0) & (b < a) & (c < b)
        eligible = extinction & out['daily_vol'].notna() & (out['daily_vol'] > 0)
        out['candidate'] = np.select([eligible & accelerating_down, eligible & accelerating_up], [1, -1], default=0).astype(int)
        # Portfolio allocation multiplies signal size by its configured 0.95.
        out['entry_size'] = (params.target_daily_vol / (0.95 * out['daily_vol'].clip(lower=0.001))).clip(0.05, 1.0).fillna(1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalExtinctionParams) -> SignalFrame:
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position = remaining = 0
        held_size = 1.0
        for i, candidate in enumerate(candidates):
            if position:
                if remaining == 0:
                    # Force an exit bar; ignore all entry signals on this bar.
                    position = 0
                else:
                    signals[i] = position
                    sizes[i] = held_size
                    remaining -= 1
            elif candidate:
                position = int(candidate)
                remaining = params.hold_bars - 1
                held_size = float(entry_sizes[i])
                signals[i] = position
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'candidate': [1, -1, 1, -1, 1, -1, 1, -1], 'entry_size': np.arange(1, 9) / 10.0}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=10)
    for hold, expected in ((1, [0, 1, 0, 1, 0, 1, 0, 1]), (2, [0, 1, 1, 0, -1, -1, 0, 1])):
        result = strategy.generate_signals(data, features, ctx, OrdinalExtinctionParams(hold_bars=hold)).data
        assert result['signal'].tolist() == expected
        assert (result['size'] > 0).all()
        if hold == 2:
            assert np.allclose(result['size'].iloc[[1, 2, 4, 5]], [0.1, 0.1, 0.4, 0.4])
    flat = strategy.indicators(data, OrdinalExtinctionParams())
    assert flat['candidate'].eq(0).all()
    assert np.isfinite(flat['entry_size']).all()
