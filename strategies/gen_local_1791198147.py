from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    acceleration_atr: float = 0.35
    trail_atr: float = 1.5

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in (self.acceleration_atr, self.trail_atr)):
            raise ValueError('Parameters must be finite and positive.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791198147'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['roc'] = close.pct_change(3, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['previous_acceleration'] = out['acceleration'].shift(1)
        out['acceleration_scaled'] = out['acceleration'] / (out['atr'] / close).replace(0, np.nan)
        out['upper'] = data['high'].rolling(5, min_periods=5).max().shift(1)
        out['lower'] = data['low'].rolling(5, min_periods=5).min().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & (out['atr'] > 0) & (close > 0)
        long_entry = valid & (close > out['upper']) & (out['roc'] > 0) & (out['previous_acceleration'] <= 0) & (out['acceleration_scaled'] >= params.acceleration_atr)
        short_entry = valid & (close < out['lower']) & (out['roc'] < 0) & (out['previous_acceleration'] >= 0) & (out['acceleration_scaled'] <= -params.acceleration_atr)
        out['entry_signal'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        desired = np.zeros(count, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry_signal'].to_numpy(dtype=int)
        side = pending = 0
        entry = best = stop_profit = 0.0
        entry_bar = eligible_bar = 0
        armed = False
        for i in range(self.warmup_bars(params) - 1, count):
            if pending:
                side = pending
                pending = 0
                entry = opens[i]
                entry_bar = i
                best = 0.0
                armed = False
                stop_profit = -2.0 * atr[i - 1]
            if side:
                if not (np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                    desired[i] = side
                    continue
                profit = side * (closes[i] - entry)
                best = max(best, profit)
                armed = armed or best >= 0.01 * entry
                if armed:
                    # Ratchet upward in favorable-distance coordinates for either direction.
                    stop_profit = max(stop_profit, 0.0, best - params.trail_atr * atr[i])
                # Stops are evaluated at closes; flattening fills at the next open.
                if profit <= stop_profit or i - entry_bar + 1 >= 4:
                    side = 0
                    eligible_bar = i + 4
                desired[i] = side
                continue
            if i >= eligible_bar and candidates[i] and i + 1 < count:
                pending = int(candidates[i])
                desired[i] = pending
        df = pd.DataFrame({'signal': desired, 'size': np.ones(count, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=36, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000000.0}, index=index)
    data.loc[index[16], 'close'] = 102.0
    data.loc[index[22], 'close'] = 98.0
    indicators = strategy.indicators(data, params)
    indicators['atr'] = 1.0
    indicators['entry_signal'] = 0
    indicators.loc[index[[15, 27]], 'entry_signal'] = 1
    indicators.loc[index[21], 'entry_signal'] = -1
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(36, dtype=int)
    expected[16:18] = 1
    expected[22:24] = -1
    expected[28:32] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert result.index.equals(data.index) and (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
