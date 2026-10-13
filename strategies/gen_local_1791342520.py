from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InventoryParams:
    inventory_bars: int = 5
    clearing_percentile: float = 0.4

    def __post_init__(self):
        if not isinstance(self.inventory_bars, int) or not 2 <= self.inventory_bars <= 6:
            raise ValueError('inventory_bars must be an integer between 2 and 6')
        if not 0.0 < self.clearing_percentile < 1.0:
            raise ValueError('clearing_percentile must be between zero and one')


class GeneratedStrategy(BaseStrategy[InventoryParams]):
    strategy_id = 'gen_local_1791342520'

    @classmethod
    def params_type(cls):
        return InventoryParams

    def warmup_bars(self, params: InventoryParams) -> int:
        return params.inventory_bars + 4

    def indicators(self, data: pd.DataFrame, params: InventoryParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        volume = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0.0)
        loss = pd.Series(0.0, index=data.index)
        inventory_volume = pd.Series(0.0, index=data.index)
        for lag in range(params.inventory_bars):
            historical_close = close.shift(lag)
            historical_volume = volume.shift(lag)
            loss = loss + historical_volume * (historical_close - close).clip(lower=0.0)
            inventory_volume = inventory_volume + historical_volume
        denominator = (inventory_volume * close).replace(0.0, np.nan)
        burden = (loss / denominator).replace([np.inf, -np.inf], np.nan)
        percentile = burden.rolling(5, min_periods=5).rank(method='average', pct=True)
        peak = close.rolling(params.inventory_bars, min_periods=params.inventory_bars).max()
        return pd.DataFrame({
            'inventory_loss_burden': burden,
            'burden_percentile': percentile,
            'local_peak': peak,
            'drawdown': (peak - close) / peak,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: InventoryParams) -> SignalFrame:
        # The same condition maintains exposure and exits when it becomes false.
        condition = (
            indicators['burden_percentile'].le(params.clearing_percentile)
            & indicators['inventory_loss_burden'].gt(0.0)
            & indicators['drawdown'].gt(0.0)
            & indicators.notna().all(axis=1)
        )
        df = pd.DataFrame(index=data.index)
        df['signal'] = condition.fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    close = np.array([100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100,
                      100.5, 101, 100.8, 100.6, 100.9, 101, 102, 101, 100], dtype=float)
    data = pd.DataFrame({
        'open': close, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': np.full(len(close), 1000.0),
    }, index=pd.date_range('2020-01-01', periods=len(close)))
    strategy = GeneratedStrategy()
    for window in (4, 5, 6):
        params = InventoryParams(inventory_bars=window)
        features = strategy.indicators(data, params)
        signals = strategy.generate_signals(data, features, None, params).data
        decision = (features['burden_percentile'].le(params.clearing_percentile)
                    & features['inventory_loss_burden'].gt(0.0)
                    & features['drawdown'].gt(0.0)
                    & features.notna().all(axis=1)).astype(int)
        pd.testing.assert_series_equal(signals['signal'],
            decision.shift(1).fillna(0).astype(int), check_names=False)
        assert signals['signal'].isin([0, 1]).all()
        assert signals['signal'].sum() > 0
        assert (signals['size'] > 0.0).all()
        assert strategy.warmup_bars(params) <= 10
        prefix = data.iloc[:13]
        pd.testing.assert_frame_equal(features.iloc[:13], strategy.indicators(prefix, params))
        prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
        pd.testing.assert_frame_equal(signals.iloc[:13], prefix_signals)
