from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SupplyBreakoutParams:
    supply_window: int = 12
    volume_window: int = 20
    pressure_window: int = 4
    pressure_threshold: float = 0.6
    position_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SupplyBreakoutParams]):
    strategy_id = 'gen_local-w2_1791317113'

    @classmethod
    def params_type(cls):
        return SupplyBreakoutParams

    @staticmethod
    def warmup_bars(params: SupplyBreakoutParams) -> int:
        return int(max(params.supply_window + 1,
                       params.volume_window + params.pressure_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SupplyBreakoutParams) -> pd.DataFrame:
        if min(params.supply_window, params.volume_window, params.pressure_window) < 1:
            raise ValueError('Lookback windows must be positive.')
        if not np.isfinite(params.pressure_threshold) or params.pressure_threshold < 0:
            raise ValueError('Pressure threshold must be finite and nonnegative.')
        if not np.isfinite(params.position_size) or not 0 < params.position_size <= 1:
            raise ValueError('Position size must be in (0, 1].')

        out = pd.DataFrame(index=data.index)
        out['volume_baseline'] = data['volume'].shift(1).rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out['close_change'] = data['close'].diff()
        out['excess_volume'] = (
            data['volume'] / out['volume_baseline'].where(out['volume_baseline'] > 0) - 1.0
        ).clip(lower=0).replace([np.inf, -np.inf], np.nan)
        out['signed_participation'] = np.sign(out['close_change']) * out['excess_volume']
        out['volume_pressure'] = out['signed_participation'].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out['bearish_high'] = data['high'].where(data['close'] < data['open'])
        out['supply_ceiling'] = out['bearish_high'].shift(1).rolling(
            params.supply_window, min_periods=1
        ).max()
        out['supply_history_count'] = data['close'].shift(1).rolling(
            params.supply_window, min_periods=1
        ).count()
        out['ready'] = (
            out['supply_history_count'].eq(params.supply_window)
            & out['supply_ceiling'].notna()
            & out['volume_pressure'].notna()
        )
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SupplyBreakoutParams) -> SignalFrame:
        # The same two-primitive predicate governs entry and continued holding.
        # Losing either agreement exits; there is no independent timed exit.
        agreement = (
            indicators['ready'].fillna(False)
            & data['close'].gt(indicators['supply_ceiling'])
            & indicators['volume_pressure'].gt(params.pressure_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df['signal'] = agreement.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.position_size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [2.0] * 5}, index=index)
    derived = pd.DataFrame({
        'ready': [True] * 5,
        'supply_ceiling': [1.0, 1.0, 3.0, 1.0, 1.0],
        'volume_pressure': [1.0, 0.0, 1.0, 1.0, np.nan],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, SupplyBreakoutParams())
    assert result.data['signal'].tolist() == [0, 1, 0, 0, 1]
    assert result.data['size'].gt(0).all()
