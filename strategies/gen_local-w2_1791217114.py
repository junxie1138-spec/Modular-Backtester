from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DisplacementParams:
    ma_window: int = 40
    memory_window: int = 126
    entry_threshold: float = 0.65


class GeneratedStrategy(BaseStrategy[DisplacementParams]):
    strategy_id = 'gen_local-w2_1791217114'

    @classmethod
    def params_type(cls):
        return DisplacementParams

    def warmup_bars(self, params: DisplacementParams) -> int:
        return params.ma_window + 18 + params.memory_window

    def indicators(self, data: pd.DataFrame, params: DisplacementParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.memory_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError('Entry threshold must be finite and positive.')
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['dispersion'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0.0, np.nan)
        out['z'] = (close - out['ma']) / out['dispersion']
        out['lagged_z'] = out['z'].shift(18)
        out['persistence'] = out['z'].rolling(params.memory_window, min_periods=params.memory_window).corr(out['lagged_z']).clip(-0.95, 0.95)
        out['anchor_drift'] = out['ma'].diff(18) / out['dispersion']
        out['forecast'] = out['anchor_drift'] + (out['persistence'] - 1.0) * out['z']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DisplacementParams) -> SignalFrame:
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError('Entry threshold must be finite and positive.')
        forecast = indicators['forecast'].to_numpy(dtype=float)
        displacement = indicators['z'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        position = 0
        entry_size = 0.20
        for i in range(len(data)):
            candidate = 0
            valid = np.isfinite(forecast[i]) and np.isfinite(displacement[i])
            if valid and abs(displacement[i]) >= 0.75:
                if forecast[i] >= params.entry_threshold:
                    candidate = 1
                elif forecast[i] <= -params.entry_threshold:
                    candidate = -1
            if position == 0:
                if candidate != 0:
                    position = candidate
                    entry_size = float(np.clip(0.35 * abs(forecast[i]) / params.entry_threshold, 0.20, 0.95))
            elif candidate == -position:
                position = 0
                entry_size = 0.20
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = DisplacementParams()
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'forecast': [2.0, 0.0, -2.0, -2.0, 0.0, 2.0], 'z': 1.0}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result['size'] > 0).all()
    weaker = derived.copy()
    weaker['forecast'] = weaker['forecast'] / 2.0
    smaller = strategy.generate_signals(data, weaker, None, params).data
    assert smaller['size'].iloc[1] < result['size'].iloc[1]
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert flat['signal'].eq(0).all()
