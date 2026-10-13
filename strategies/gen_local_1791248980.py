from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 12
    complexity_drop: float = 0.04
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791248980'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, 2 * params.pattern_window + 3, 15))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.pattern_window < 3 or params.hold_bars not in (1, 2):
            raise ValueError('pattern_window must be >= 3 and hold_bars must be 1 or 2')
        if not np.isfinite(params.complexity_drop) or params.complexity_drop <= 0:
            raise ValueError('complexity_drop must be finite and positive')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')
        close = data['close']
        triples = pd.concat([close.shift(2), close.shift(1), close], axis=1)
        valid = np.isfinite(triples).all(axis=1)
        # Stable ordering assigns equal closes by their chronological position.
        order = np.argsort(triples.to_numpy(), axis=1, kind='stable')
        pattern = pd.Series(order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2], index=data.index).where(valid)
        out = pd.DataFrame(index=data.index)
        out['pattern'] = pattern
        columns = []
        for code in (5, 7, 11, 15, 19, 21):
            name = f'p_{code}'
            out[name] = pattern.eq(code).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            columns.append(name)
        probabilities = out[columns]
        safe_probabilities = probabilities.where(probabilities > 0, 1.0)
        out['entropy'] = -(probabilities * np.log(safe_probabilities)).sum(axis=1, min_count=6) / np.log(6.0)
        out['disequilibrium'] = probabilities.sub(1.0 / 6.0).pow(2).sum(axis=1, min_count=6) / (5.0 / 6.0)
        out['complexity'] = out['entropy'] * out['disequilibrium']
        out['previous_complexity'] = out['complexity'].shift(params.pattern_window)
        out['complexity_loss'] = out['previous_complexity'] - out['complexity']
        out['monotonic_mass'] = out['p_5'] + out['p_21']
        out['direction_balance'] = out['p_5'] - out['p_21']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        ready = np.isfinite(out[['complexity_loss', 'monotonic_mass', 'direction_balance', 'ma200', 'atr']]).all(axis=1) & np.isfinite(close) & out['atr'].gt(0)
        transition = ready & out['complexity_loss'].gt(params.complexity_drop) & out['monotonic_mass'].ge(0.5) & out['direction_balance'].abs().ge(0.25)
        out['transition'] = transition.astype(int)
        out['event'] = (transition & ~transition.shift(1, fill_value=False)).astype(int)
        out['entry'] = np.select([out['event'].eq(1) & out['direction_balance'].gt(0) & close.gt(out['ma200']), out['event'].eq(1) & out['direction_balance'].lt(0) & close.lt(out['ma200'])], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        position = 0
        age = 0
        high_water = -np.inf
        low_water = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                # This bar follows the entry decision and is actually held.
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                trailing_floor = max(trailing_floor, high_water - params.trail_atr * atr)
                short_ceiling = min(short_ceiling, low_water + params.trail_atr * atr)
                # Both sides obey the requested high-water drawdown exit.
                # Shorts additionally have a mirrored adverse-move ATR trail.
                stopped = close < trailing_floor or (position == -1 and close > short_ceiling)
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                # Initialize watermarks on the first held bar, after the fill.
                high_water = -np.inf
                low_water = np.inf
                trailing_floor = -np.inf
                short_ceiling = np.inf
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': [100.0, 100.0, 103.0, 101.0, 100.0, 100.0, 101.0, 100.0], 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'entry': [0, 1, 0, 0, 0, -1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, -1, -1]
    assert result['size'].gt(0).all()
    for n in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:n], derived.iloc[:n], None, params).data
        assert prefix.equals(result.iloc[:n])
    params.hold_bars = 1
    one_day = strategy.generate_signals(data, derived, None, params).data
    assert one_day['signal'].tolist() == [0, 0, 1, 0, 0, 0, -1, 0]
