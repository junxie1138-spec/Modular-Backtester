from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 18
    complexity_floor: float = 0.06
    efficiency_window: int = 10
    efficiency_floor: float = 0.45
    atr_window: int = 10
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791314489'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.pattern_window + 3,
                       params.efficiency_window + 1,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.pattern_window, params.efficiency_window, params.atr_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if params.trail_atr <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError('Require positive trail_atr and hold_bars between 3 and 5.')
        close = data['close']
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns retain chronological order within the ordinal pattern.
        pattern = ((a > b).astype(int) * 4
                   + (a > c).astype(int) * 2
                   + (b > c).astype(int)).where(valid)
        out = pd.DataFrame(index=data.index)
        out['return'] = returns
        out['ordinal_pattern'] = pattern
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f'pattern_probability_{code}'
            out[name] = pattern.eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        entropy_terms = probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
        out['ordinal_entropy'] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out['ordinal_disequilibrium'] = ((probabilities - 1.0 / 6.0) ** 2).sum(
            axis=1, min_count=6
        ) / (5.0 / 6.0)
        out['ordinal_complexity'] = out['ordinal_entropy'] * out['ordinal_disequilibrium']
        out['price_change'] = close.diff()
        out['net_displacement'] = out['price_change'].rolling(params.efficiency_window).sum()
        out['path_length'] = out['price_change'].abs().rolling(params.efficiency_window).sum()
        out['directional_efficiency'] = out['net_displacement'] / out['path_length'].replace(0, np.nan)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators[['ordinal_complexity', 'directional_efficiency', 'atr']].notna().all(axis=1)
        entry = (ready
                 & indicators['ordinal_complexity'].ge(params.complexity_floor)
                 & indicators['directional_efficiency'].ge(params.efficiency_floor)
                 & indicators['atr'].gt(0))
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = entry.fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    active = False
                    continue
                # The first iteration after entry is the first actual holding bar.
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
            elif entries[i] and np.isfinite(closes[i]):
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 102.0, 105.0, 104.0, 103.0, 104.0]}, index=index)
    derived = pd.DataFrame({
        'ordinal_complexity': [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        'directional_efficiency': [0.8] * 6,
        'atr': [1.0, 1.0, 1.0, 10.0, 10.0, 1.0]
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, Params(hold_bars=5))
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0], 'Shift or ratcheting stop failed.'
