from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_window: int = 10
    acceleration_window: int = 5
    noise_window: int = 20
    atr_window: int = 14
    acceleration_threshold: float = 0.35
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = 'gen_local-w2_1791208696'

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(
            params.roc_window + params.acceleration_window + 1,
            params.noise_window + 1,
            params.atr_window + 1,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        windows = (params.roc_window, params.acceleration_window,
                   params.noise_window, params.atr_window, params.hold_bars)
        if any(w < 1 for w in windows) or params.acceleration_window < 2:
            raise ValueError('Lookbacks must be positive; acceleration_window must be at least two.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be positive and finite.')
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold < 0:
            raise ValueError('acceleration_threshold must be nonnegative and finite.')

        out = pd.DataFrame(index=data.index)
        log_close = np.log(data['close'].where(data['close'] > 0))
        out['log_return'] = log_close.diff()
        out['roc'] = log_close.diff(params.roc_window)
        out['roc_increment'] = out['log_return'] - out['log_return'].shift(params.roc_window)
        increments = out['roc_increment'].rolling(
            params.acceleration_window, min_periods=params.acceleration_window)
        out['acceleration'] = increments.sum()
        out['largest_contribution'] = increments.max().clip(lower=0.0)
        out['robust_acceleration'] = out['acceleration'] - out['largest_contribution']
        out['noise'] = out['log_return'].rolling(
            params.noise_window, min_periods=params.noise_window).std(ddof=0)
        denominator = out['noise'].replace(0.0, np.nan) * np.sqrt(2.0 * params.acceleration_window)
        out['acceleration_score'] = out['robust_acceleration'] / denominator

        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['candidate'] = (
            (out['roc'] > 0.0)
            & (out['acceleration_score'] > params.acceleration_threshold)
            & (out['atr'] > 0.0)
        ).fillna(False)
        out['confirmed'] = out['candidate'] & out['candidate'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        active = False
        stop = np.nan
        entry_bar = -1

        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open. Freeze its known ATR.
            if not active and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]

            if active:
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                    # Exit decisions also execute on the following bar.
                    continue
                decisions[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = AccelerationParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0, 'high': 111.0, 'low': 93.0,
        'close': [100.0, 100.0, 100.0, 110.0, 100.0, 94.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0],
        'confirmed': [False, False, True, True, True, True, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
