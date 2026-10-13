from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeChangeParams:
    baseline_bars: int = 20
    reaction_bars: int = 3
    atr_bars: int = 14
    expansion_ratio: float = 1.35
    migration_atr: float = 0.40
    pressure_threshold: float = 0.20
    stop_atr: float = 1.50
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeChangeParams]):
    strategy_id = 'gen_local_1791243402'

    @classmethod
    def params_type(cls) -> type[RangeChangeParams]:
        return RangeChangeParams

    def warmup_bars(self, params: RangeChangeParams) -> int:
        return int(max(params.baseline_bars + params.reaction_bars,
                       params.atr_bars + 1, params.reaction_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RangeChangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out['relative_range'] = out['range'] / previous_close.where(previous_close > 0)
        out['recent_range'] = out['relative_range'].rolling(params.reaction_bars).mean()
        out['baseline_range'] = out['relative_range'].shift(params.reaction_bars).rolling(params.baseline_bars).mean()
        out['expansion'] = out['recent_range'] / out['baseline_range'].where(out['baseline_range'] > 0)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['migration'] = (out['midpoint'] - out['midpoint'].shift(params.reaction_bars)) / out['atr'].where(out['atr'] > 0)
        out['volume'] = data['volume'].clip(lower=0)
        out['body_pressure'] = ((data['close'] - data['open']) / out['range'].where(out['range'] > 0)).clip(-1, 1).fillna(0)
        out['signed_volume'] = out['body_pressure'] * out['volume']
        out['volume_sum'] = out['volume'].rolling(params.reaction_bars).sum()
        out['pressure'] = out['signed_volume'].rolling(params.reaction_bars).sum() / out['volume_sum'].where(out['volume_sum'] > 0)
        ready = np.isfinite(out[['atr', 'expansion', 'migration', 'pressure']]).all(axis=1) & out['atr'].gt(0)
        change = out['expansion'].ge(params.expansion_ratio)
        onset = change & ~change.shift(1, fill_value=False)
        long_entry = ready & onset & out['migration'].ge(params.migration_atr) & out['pressure'].ge(params.pressure_threshold)
        short_entry = ready & onset & out['migration'].le(-params.migration_atr) & out['pressure'].le(-params.pressure_threshold)
        out['candidate'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeChangeParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        stop_distance = np.nan
        age = 0
        for i in range(len(data)):
            if position != 0:
                # A decision on the preceding close fills at today's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    stop_distance = params.stop_atr * atr[i - 1]
                    age = 0
                age += 1
                valid = np.isfinite(entry_price) and np.isfinite(stop_distance) and np.isfinite(closes[i])
                stopped = valid and position * (closes[i] - entry_price) < -stop_distance
                if not valid or stopped or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
            elif candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                position = int(candidates[i])
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RangeChangeParams()
    index = pd.date_range('2020-01-01', periods=8)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 94.0,
                             'close': 100.0, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 2.0, 'candidate': [direction, 0, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        data.loc[index[3], 'close'] = 100.0 - direction * 4.0
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0, 0, 0, 0]
        assert result['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['candidate'].eq(0).all()
    assert strategy.warmup_bars(params) >= params.baseline_bars + params.reaction_bars


if __name__ == '__main__':
    _self_check()
