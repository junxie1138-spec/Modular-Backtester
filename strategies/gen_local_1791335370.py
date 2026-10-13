from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalStrengthParams:
    pattern_window: int = 12
    direction_window: int = 4
    max_entropy: float = 0.80
    ordinal_bias: float = 0.15
    dm_threshold: float = 0.25
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalStrengthParams]):
    strategy_id = 'gen_local_1791335370'

    @classmethod
    def params_type(cls) -> type[OrdinalStrengthParams]:
        return OrdinalStrengthParams

    def warmup_bars(self, params: OrdinalStrengthParams) -> int:
        return int(max(params.pattern_window + 2, params.direction_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalStrengthParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.direction_window < 1:
            raise ValueError('Lookback windows must be positive; pattern_window must be at least 3.')
        if not (0.0 <= params.max_entropy <= 1.0 and 0.0 < params.ordinal_bias <= 1.0 and 0.0 < params.dm_threshold <= 1.0):
            raise ValueError('Invalid entropy or directional thresholds.')
        if params.profit_target <= 0.0 or not np.isfinite(params.profit_target) or params.max_hold_bars not in (1, 2):
            raise ValueError('Use a positive finite profit target and a one- or two-bar time stop.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & (data['close'] > 0.0))
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & b.ne(c) & a.ne(c)
        patterns = (
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        )
        for k, pattern in enumerate(patterns):
            out[f'pattern_frequency_{k}'] = pattern.astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
        frequencies = out[[f'pattern_frequency_{k}' for k in range(6)]]
        # Zero-probability categories contribute zero; incomplete windows stay NaN.
        terms = frequencies * np.log(frequencies.clip(lower=1e-12))
        out['ordinal_entropy'] = -terms.sum(axis=1, min_count=6) / np.log(6.0)
        out['ordinal_direction'] = out['pattern_frequency_0'] - out['pattern_frequency_5']
        high = data['high'].where(np.isfinite(data['high']) & (data['high'] > 0.0))
        low = data['low'].where(np.isfinite(data['low']) & (data['low'] > 0.0))
        up = high.diff()
        down = -low.diff()
        dm_valid = up.notna() & down.notna() & high.ge(low) & high.shift(1).ge(low.shift(1))
        out['up_dm'] = up.where((up > down) & (up > 0.0), 0.0).where(dm_valid)
        out['down_dm'] = down.where((down > up) & (down > 0.0), 0.0).where(dm_valid)
        out['up_dm_sum'] = out['up_dm'].rolling(params.direction_window, min_periods=params.direction_window).sum()
        out['down_dm_sum'] = out['down_dm'].rolling(params.direction_window, min_periods=params.direction_window).sum()
        total = out['up_dm_sum'] + out['down_dm_sum']
        out['range_direction'] = (out['up_dm_sum'] - out['down_dm_sum']) / total.where(total > 0.0)
        ordered = out['ordinal_entropy'].le(params.max_entropy)
        long_entry = ordered & out['ordinal_direction'].ge(params.ordinal_bias) & out['range_direction'].ge(params.dm_threshold)
        short_entry = ordered & out['ordinal_direction'].le(-params.ordinal_bias) & out['range_direction'].le(-params.dm_threshold)
        out['entry_direction'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalStrengthParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2) or params.profit_target <= 0.0 or not np.isfinite(params.profit_target):
            raise ValueError('Invalid exit parameters.')
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close scheduled entry; its fill proxy becomes known now.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price else 0.0
                if not valid_price or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    # Force a flat decision on exit; do not reverse or reenter here.
                    continue
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': np.ones(len(data), dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = OrdinalStrengthParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_direction': 1}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    assert result['size'].gt(0.0).all()
    data['close'] = 102.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 0, 1]
    indicators['entry_direction'] = -1
    data['close'] = 98.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, 0, -1, 0, -1]
    index = pd.date_range('2020-01-01', periods=30)
    close = np.arange(100.0, 130.0)
    trend = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(trend, params)
    assert derived['entry_direction'].iloc[:params.pattern_window + 1].eq(0).all()
    assert derived['entry_direction'].iloc[-1] == 1
    pd.testing.assert_frame_equal(derived.iloc[:20], strategy.indicators(trend.iloc[:20], params))
    full = strategy.generate_signals(trend, derived, None, params).data
    prefix = strategy.generate_signals(trend.iloc[:20], derived.iloc[:20], None, params).data
    pd.testing.assert_frame_equal(full.iloc[:20], prefix)


if __name__ == '__main__':
    _self_check()
