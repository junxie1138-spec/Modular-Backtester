from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecurrenceParams:
    roc_bars: int = 3
    correlation_bars: int = 32
    recurrence_threshold: float = 0.20
    dominance_margin: float = 0.10
    acceleration_threshold: float = 0.25
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if self.roc_bars < 1 or self.correlation_bars < 3 or self.atr_bars < 1:
            raise ValueError('Lookback lengths must be positive; correlation requires at least three bars.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        if not 0 <= self.recurrence_threshold < 1:
            raise ValueError('recurrence_threshold must be in [0, 1).')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        if not np.isfinite(self.dominance_margin) or self.dominance_margin < 0:
            raise ValueError('dominance_margin must be finite and nonnegative.')
        if not np.isfinite(self.acceleration_threshold) or self.acceleration_threshold < 0:
            raise ValueError('acceleration_threshold must be finite and nonnegative.')


class GeneratedStrategy(BaseStrategy[RecurrenceParams]):
    strategy_id = 'gen_local-w2_1791308479'

    @classmethod
    def params_type(cls):
        return RecurrenceParams

    def warmup_bars(self, params: RecurrenceParams) -> int:
        return int(max(params.roc_bars + params.correlation_bars + 3, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RecurrenceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['acceleration_lag1'] = out['acceleration'].shift(1)
        out['acceleration_lag2'] = out['acceleration'].shift(2)
        window = out['acceleration'].rolling(params.correlation_bars, min_periods=params.correlation_bars)
        out['acceleration_scale'] = window.std(ddof=0).replace(0.0, np.nan)
        out['normalized_acceleration'] = out['acceleration'] / out['acceleration_scale']
        out['correlation_lag1'] = window.corr(out['acceleration_lag1']).clip(-1.0, 1.0)
        out['correlation_lag2'] = window.corr(out['acceleration_lag2']).clip(-1.0, 1.0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['recurrence_advantage'] = out['correlation_lag2'] - out['correlation_lag1'].abs()
        eligible = (
            out['correlation_lag2'].gt(params.recurrence_threshold)
            & out['recurrence_advantage'].gt(params.dominance_margin)
            & out['atr'].gt(0.0)
        )
        out['candidate'] = np.select([
            eligible & out['normalized_acceleration'].gt(params.acceleration_threshold),
            eligible & out['normalized_acceleration'].lt(-params.acceleration_threshold),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecurrenceParams) -> SignalFrame:
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        entry_bar = -1
        stop_price = np.nan
        entry_atr = np.nan

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    position = 0
                else:
                    entry_bar = i
                    # Freeze the stop using the fill open and decision-bar ATR.
                    stop_price = opens[i] - position * params.stop_atr * entry_atr

            if position:
                stop_hit = np.isfinite(closes[i]) and position * (closes[i] - stop_price) <= 0.0
                time_exit = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or time_exit:
                    position = 0
            elif (
                i > 0
                and candidates[i] != 0
                and candidates[i] == candidates[i - 1]
                and np.isfinite(closes[i])
                and closes[i] > 0
                and np.isfinite(atr[i])
                and atr[i] > 0
            ):
                position = int(candidates[i])
                entry_atr = float(atr[i])
                pending_entry = True

            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0, 'high': 104.0, 'low': 96.0,
        'close': [100.0, 100.0, 97.0, 100.0, 100.0, 103.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    features = pd.DataFrame({'candidate': [1, 1, 0, -1, -1, 0, 0], 'atr': 1.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, features, None, RecurrenceParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, -1, 0]
    data['close'] = 100.0
    features['candidate'] = 1
    result = strategy.generate_signals(data, features, None, RecurrenceParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0]
