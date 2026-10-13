from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    lookback: int = 6
    migration_threshold: float = 0.35
    expansion_threshold: float = 1.15
    profit_target: float = 0.04
    max_hold_bars: int = 18

    def __post_init__(self):
        if not 1 <= self.lookback <= 9:
            raise ValueError('lookback must be between 1 and 9')
        if not 0.0 < self.migration_threshold <= 1.0:
            raise ValueError('migration_threshold must be in (0, 1]')
        if not np.isfinite(self.expansion_threshold) or self.expansion_threshold <= 0:
            raise ValueError('expansion_threshold must be positive and finite')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite')
        if self.max_hold_bars < 1:
            raise ValueError('max_hold_bars must be positive')


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = 'gen_local_1791256071'

    @classmethod
    def params_type(cls):
        return RangeMigrationParams

    @staticmethod
    def warmup_bars(params: RangeMigrationParams) -> int:
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        out['high_move'] = data['high'].diff()
        out['low_move'] = data['low'].diff()
        out['shared_up'] = np.minimum(out['high_move'].clip(lower=0), out['low_move'].clip(lower=0))
        out['shared_down'] = np.minimum((-out['high_move']).clip(lower=0), (-out['low_move']).clip(lower=0))
        out['boundary_motion'] = np.maximum(out['high_move'].abs(), out['low_move'].abs())
        out['shared_net'] = out['shared_up'] - out['shared_down']
        out['net_sum'] = out['shared_net'].rolling(params.lookback, min_periods=params.lookback).sum()
        out['motion_sum'] = out['boundary_motion'].rolling(params.lookback, min_periods=params.lookback).sum()
        out['migration'] = out['net_sum'] / out['motion_sum'].replace(0, np.nan)
        out['prior_range_mean'] = out['range'].shift(1).rolling(params.lookback, min_periods=params.lookback).mean()
        out['expansion'] = out['range'] / out['prior_range_mean'].replace(0, np.nan)
        ready = out['migration'].notna() & out['expansion'].notna()
        expanding = out['expansion'].ge(params.expansion_threshold)
        out['setup'] = np.select(
            [ready & expanding & out['migration'].ge(params.migration_threshold),
             ready & expanding & out['migration'].le(-params.migration_threshold)],
            [1, -1], default=0
        ).astype(int)
        out['entry'] = out['setup'].where(out['setup'].ne(out['setup'].shift(1).fillna(0)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeMigrationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's observed open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                # ponytail: open-price fill proxy; use recorded fills if the contract exposes them.
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                timed_out = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or target_hit or timed_out:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
            decisions[i] = position

        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RangeMigrationParams()
    index = pd.date_range('2020-01-01', periods=22, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.warmup_bars(params) <= 10
    assert derived['entry'].eq(0).all()
    for direction in (1, -1):
        entries = pd.DataFrame({'entry': np.zeros(len(data), dtype=int)}, index=index)
        entries.iloc[0, 0] = direction
        result = GeneratedStrategy.generate_signals(data, entries, None, params).data
        assert result['signal'].iloc[0] == 0
        assert result['signal'].iloc[1:19].eq(direction).all()
        assert result['signal'].iloc[19:].eq(0).all()
        target_data = data.copy()
        target_data.loc[index[2], 'close'] = 100.0 * (1.0 + direction * 0.05)
        target_result = GeneratedStrategy.generate_signals(target_data, entries, None, params).data
        assert target_result['signal'].iloc[1:3].eq(direction).all()
        assert target_result['signal'].iloc[3:].eq(0).all()
        assert target_result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
