from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    ma_window: int = 20
    volume_window: int = 20
    pressure_window: int = 3
    percentile_window: int = 126
    entry_percentile: float = 0.90
    volatility_window: int = 20
    target_volatility: float = 0.12
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.ma_window, self.volume_window, self.volatility_window, self.percentile_window) < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if self.pressure_window < 1 or not 3 <= self.hold_bars <= 5:
            raise ValueError('Invalid pressure window or holding horizon.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('Entry percentile must lie between zero and one.')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError('Target volatility must be finite and positive.')


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = 'gen_local-w2_1791345625'

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(max(
            max(params.ma_window, params.volume_window)
            + params.pressure_window + params.percentile_window + 1,
            params.volatility_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] > 0)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].replace(0.0, np.nan)
        out['volume_ma'] = volume.rolling(params.volume_window).mean()
        out['relative_volume'] = volume / out['volume_ma'].replace(0.0, np.nan)
        out['capacity_stretch'] = out['z'] / out['relative_volume'].clip(lower=0.25)
        out['pressure'] = out['capacity_stretch'].rolling(params.pressure_window).sum()
        out['absolute_pressure'] = out['pressure'].abs()
        out['threshold'] = out['absolute_pressure'].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        out['returns'] = close.pct_change(fill_method=None)
        out['volatility'] = out['returns'].rolling(params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        crossed = (
            (out['absolute_pressure'] > out['threshold'])
            & (out['absolute_pressure'].shift(1) <= out['threshold'].shift(1))
        )
        valid = (
            crossed
            & (out['pressure'] * out['z'] > 0)
            & out['volatility'].gt(0)
        )
        out['entry'] = np.where(valid, -np.sign(out['pressure'].fillna(0.0)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        volatility = indicators['volatility'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        direction = 0
        exit_bar = -1
        held_size = 0.95
        for i in range(len(data)):
            if direction:
                if i == exit_bar:
                    direction = 0
            elif entries[i] and np.isfinite(volatility[i]) and volatility[i] > 0:
                direction = int(entries[i])
                exit_bar = i + params.hold_bars
                held_size = float(np.clip(params.target_volatility / volatility[i], 0.05, 0.95))
            signals[i] = direction
            if direction:
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = np.zeros(12, dtype=int)
    entries[[0, 1, 2, 4, 5]] = [1, -1, -1, 1, -1]
    ind = pd.DataFrame({'entry': entries, 'volatility': 0.20}, index=index)
    result = GeneratedStrategy().generate_signals(data, ind, None, CapacityParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert np.allclose(result.loc[result['signal'] != 0, 'size'], 0.60)
    assert result['size'].gt(0).all()
