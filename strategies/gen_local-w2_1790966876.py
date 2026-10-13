from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalScaleParams:
    pattern_window: int = 18
    target_vol: float = 0.12

    def __post_init__(self):
        if not isinstance(self.pattern_window, int) or isinstance(self.pattern_window, bool) or self.pattern_window < 6:
            raise ValueError('pattern_window must be an integer >= 6')
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError('target_vol must be positive and finite')


class GeneratedStrategy(BaseStrategy[OrdinalScaleParams]):
    strategy_id = 'gen_local-w2_1790966876'

    @classmethod
    def params_type(cls):
        return OrdinalScaleParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.pattern_window + 6, 21))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volatility'] = out['return'].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['momentum'] = close / close.shift(6) - 1.0
        for label, lag in (('daily', 1), ('coarse', 3)):
            oldest = close.shift(2 * lag)
            middle = close.shift(lag)
            valid = oldest.notna() & middle.notna() & close.notna()
            # Equal prices retain chronological order, giving deterministic ranks.
            code = 4 * (oldest > middle).astype(int) + 2 * (oldest > close).astype(int) + (middle > close).astype(int)
            out[label + '_pattern'] = code.where(valid)
            frequency_columns = []
            for motif in (0, 1, 3, 4, 6, 7):
                name = label + '_count_' + str(motif)
                out[name] = out[label + '_pattern'].eq(motif).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).sum()
                frequency_columns.append(name)
            counts = out[frequency_columns]
            out[label + '_diversity'] = counts.gt(0).sum(axis=1).astype(float).where(counts.notna().all(axis=1))
        out['diversity_gap'] = out['daily_diversity'] - out['coarse_diversity']
        out['target_size'] = (params.target_vol / out['volatility'].where(out['volatility'] > 0)).clip(lower=0.01, upper=0.95).fillna(0.01)
        ready = out[['atr', 'volatility', 'momentum', 'daily_diversity', 'coarse_diversity']].notna().all(axis=1)
        out['entry_ready'] = (
            ready & (out['atr'] > 0) & (out['volatility'] > 0)
            & (out['momentum'] > 0)
            & (out['daily_diversity'] >= 5)
            & (out['coarse_diversity'] <= 3)
            & (out['diversity_gap'] >= 2)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        stops = np.full(n, np.nan)
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(n):
            row = data.iloc[i]
            atr = float(indicators['atr'].iloc[i])
            valid = np.isfinite(row[['open', 'high', 'low', 'close']].to_numpy(dtype=float)).all() and np.isfinite(atr) and atr > 0
            if not valid:
                decisions[i] = int(active)
                stops[i] = stop if active else np.nan
                continue
            # Yesterday's decision fills at today's open; no future open is read.
            if not active and i > 0 and decisions[i - 1] == 1:
                entry = float(row['open'])
                if entry <= 0:
                    continue
                prior_atr = float(indicators['atr'].iloc[i - 1])
                stop = entry - 2.0 * prior_atr
                peak = entry
                active = True
                armed = False
                age = 0
            if active:
                age += 1
                stops[i] = stop
                # A breach observed today schedules liquidation at tomorrow's open.
                if row['low'] <= stop:
                    active = False
                    continue
                peak = max(peak, float(row['high']))
                armed = armed or peak >= entry * 1.02
                if armed:
                    stop = max(stop, entry, peak - 2.0 * atr)
                stops[i] = stop
                # Newly raised stops are checked at the close, avoiding unknown intrabar ordering.
                if row['close'] <= stop or age >= 10:
                    active = False
                    continue
                decisions[i] = 1
            elif bool(indicators['entry_ready'].iloc[i]):
                decisions[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['size'] = indicators['target_size'].shift(1).fillna(0.01).clip(lower=0.01, upper=0.95).astype(float)
        df['stop'] = stops
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.iloc[2, data.columns.get_loc('high')] = 104.0
    data.iloc[2, data.columns.get_loc('close')] = 103.0
    data.iloc[3, data.columns.get_loc('high')] = 103.0
    data.iloc[3, data.columns.get_loc('low')] = 101.0
    data.iloc[3, data.columns.get_loc('close')] = 101.0
    indicators = pd.DataFrame({'atr': 1.0, 'entry_ready': True, 'target_size': 0.5}, index=index)
    params = OrdinalScaleParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1]
    assert result['stop'].iloc[2] == 102.0
    assert result['stop'].iloc[3] >= result['stop'].iloc[2]
    assert (result['size'] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) == 24
    calculated = GeneratedStrategy.indicators(data, params)
    assert not calculated['entry_ready'].any()


if __name__ == '__main__':
    _self_check()
