from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownPressureParams:
    peak_window: int = 63
    decision_window: int = 18
    minimum_depth: float = 0.03
    repair_fraction: float = 0.25
    pressure_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[DrawdownPressureParams]):
    strategy_id = 'gen_local-w2_1791237553'

    @classmethod
    def params_type(cls) -> type[DrawdownPressureParams]:
        return DrawdownPressureParams

    def warmup_bars(self, params: DrawdownPressureParams) -> int:
        return int(params.peak_window + params.decision_window)

    def indicators(self, data: pd.DataFrame, params: DrawdownPressureParams) -> pd.DataFrame:
        if params.peak_window < 1 or params.decision_window < 1:
            raise ValueError('Lookback windows must be positive.')
        if not 0.0 < params.minimum_depth < 1.0:
            raise ValueError('minimum_depth must be between zero and one.')
        if not 0.0 < params.repair_fraction < 1.0:
            raise ValueError('repair_fraction must be between zero and one.')
        if not 0.0 < params.pressure_threshold < 1.0:
            raise ValueError('pressure_threshold must be between zero and one.')
        close = data['close'].where(data['close'] > 0.0)
        anchor = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(params.decision_window)
        starting_depth = 1.0 - close.shift(params.decision_window) / anchor
        current_depth = 1.0 - close / anchor
        price_range = (data['high'] - data['low']).where(data['high'] >= data['low'])
        close_location = ((2.0 * close - data['high'] - data['low']) / price_range.where(price_range > 0.0)).clip(-1.0, 1.0)
        close_location = close_location.mask(price_range.eq(0.0), 0.0)
        volume = data['volume'].where(data['volume'] >= 0.0)
        usable_volume = volume.where(close_location.notna())
        weighted_location = close_location * usable_volume
        pressure = weighted_location.rolling(params.decision_window, min_periods=params.decision_window).sum() / usable_volume.rolling(params.decision_window, min_periods=params.decision_window).sum().replace(0.0, np.nan)
        return pd.DataFrame({
            'anchor': anchor,
            'starting_depth': starting_depth,
            'current_depth': current_depth,
            'price_range': price_range,
            'close_location': close_location,
            'usable_volume': usable_volume,
            'weighted_location': weighted_location,
            'pressure': pressure,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownPressureParams) -> SignalFrame:
        start = indicators['starting_depth']
        current = indicators['current_depth']
        pressure = indicators['pressure']
        valid = start.notna() & current.notna() & pressure.notna()
        eligible = valid & start.ge(params.minimum_depth)
        long_entry = (eligible & current.le(start * (1.0 - params.repair_fraction)) & pressure.ge(params.pressure_threshold)).to_numpy(dtype=bool)
        short_entry = (eligible & current.ge(start * (1.0 + params.repair_fraction)) & pressure.le(-params.pressure_threshold)).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        position = 0
        for i in range(len(data)):
            if position == 1:
                if short_entry[i]:
                    position = 0
            elif position == -1:
                if long_entry[i]:
                    position = 0
            elif long_entry[i]:
                position = 1
            elif short_entry[i]:
                position = -1
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'starting_depth': [0.10] * 7,
        'current_depth': [0.05, 0.10, 0.15, 0.15, 0.05, 0.05, np.nan],
        'pressure': [0.20, 0.00, -0.20, -0.20, 0.20, 0.20, np.nan],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, DrawdownPressureParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, 0, 1]
    assert result.data['size'].gt(0.0).all()
    disagreement = indicators.copy()
    disagreement['pressure'] = -disagreement['pressure']
    result = GeneratedStrategy.generate_signals(None, data, disagreement, None, DrawdownPressureParams())
    assert result.data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
