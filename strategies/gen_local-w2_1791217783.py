from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    acceleration_threshold: float = 0.4
    volume_ceiling: float = 1.0
    profit_target: float = 0.006
    max_hold_bars: int = 2

    def __post_init__(self):
        values = (self.acceleration_threshold, self.volume_ceiling, self.profit_target)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds must be finite and positive.')
        if self.volume_ceiling < 0.65:
            raise ValueError('volume_ceiling must be at least 0.65.')
        if self.max_hold_bars not in (1, 2):
            raise ValueError('max_hold_bars must be 1 or 2.')


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = 'gen_local-w2_1791217783'

    @classmethod
    def params_type(cls):
        return MomentumParams

    @staticmethod
    def warmup_bars(params: MomentumParams) -> int:
        return 7

    @staticmethod
    def indicators(data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] > 0)
        out = pd.DataFrame(index=data.index)
        out['return_1'] = close.pct_change(fill_method=None)
        out['roc_2'] = close.pct_change(periods=2, fill_method=None)
        out['acceleration'] = out['roc_2'].diff()
        out['return_scale'] = out['return_1'].abs().rolling(6, min_periods=6).mean()
        out['acceleration_score'] = out['acceleration'] / out['return_scale'].where(out['return_scale'] > 0)
        out['volume_capacity'] = volume.shift(1).rolling(5, min_periods=5).mean()
        out['volume_ratio'] = volume / out['volume_capacity']
        span = data['high'] - data['low']
        out['close_location'] = (close - data['low']) / span.where(span > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        capacity_gate = out['volume_ratio'].between(0.65, params.volume_ceiling)
        long_entry = ready & capacity_gate & (out['roc_2'] > 0) & (out['acceleration_score'] >= params.acceleration_threshold) & (out['close_location'] >= 0.65)
        short_entry = ready & capacity_gate & (out['roc_2'] < 0) & (out['acceleration_score'] <= -params.acceleration_threshold) & (out['close_location'] <= 0.35)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        candidates = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        entry_price = 0.0
        age = 0
        for i in range(len(data)):
            position = decisions[i - 1] if i else 0
            if position:
                # Yesterday's close decision enters at today's open.
                if i == 1 or decisions[i - 2] == 0:
                    entry_price = opens[i]
                    age = 0
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else -np.inf
                # Close-observed targets and time stops exit at the next open.
                exit_now = not valid_entry or gain >= params.profit_target or age >= params.max_hold_bars
                decisions[i] = 0 if exit_now else position
            else:
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                decisions[i] = candidates[i] if valid_close else 0
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    sample = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': [100, 101, 100, 100, 99, 100, 100], 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0, 0]}, index=index)
    for target, hold, expected in [(0.5, 2, [0, 1, 1, 0, -1, -1, 0]), (0.006, 2, [0, 1, 0, 0, -1, 0, 0]), (0.5, 1, [0, 1, 0, 0, -1, 0, 0])]:
        params = MomentumParams(profit_target=target, max_hold_bars=hold)
        result = GeneratedStrategy.generate_signals(sample, entries, None, params)
        assert result.data['signal'].tolist() == expected
    derived = GeneratedStrategy.indicators(sample, MomentumParams())
    assert np.isfinite(derived.to_numpy()).all()
    assert GeneratedStrategy.warmup_bars(MomentumParams()) <= 10
