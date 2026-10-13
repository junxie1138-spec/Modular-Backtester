from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    trend_window: int = 4
    shock_threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1790968809"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(int(params.trend_window) + 2, 12)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        n = int(params.trend_window)
        if n < 2 or not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0:
            raise ValueError("trend_window must be >= 2 and shock_threshold must be positive")
        prices = data[['open', 'high', 'low', 'close']].astype(float)
        valid = (
            np.isfinite(prices).all(axis=1)
            & prices.gt(0).all(axis=1)
            & prices['high'].ge(prices[['open', 'close', 'low']].max(axis=1))
            & prices['low'].le(prices[['open', 'close', 'high']].min(axis=1))
            & np.isfinite(data['volume'])
            & data['volume'].gt(0)
        )
        close = np.log(prices['close'].where(valid))
        high = np.log(prices['high'].where(valid))
        low = np.log(prices['low'].where(valid))
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        slope = sum((center - lag) * close.shift(lag) for lag in range(n)) / denominator
        level = close.rolling(n, min_periods=n).mean() + center * slope
        forecast = (level + slope).shift(1)
        scale = (high - low).rolling(10, min_periods=10).mean().shift(1)
        scale = scale.where(scale > 0)
        innovation = (close - forecast) / scale
        drawdown = (high.rolling(10, min_periods=10).max().shift(1) - close) / scale
        drawup = (close - low.rolling(10, min_periods=10).min().shift(1)) / scale
        out = pd.DataFrame(index=data.index)
        out['latent_slope'] = slope
        out['latent_level'] = level
        out['forecast'] = forecast
        out['scale'] = scale
        out['innovation'] = innovation
        out['previous_innovation'] = innovation.shift(1)
        out['drawdown'] = drawdown
        out['drawup'] = drawup
        out['higher_low'] = low.gt(low.shift(1))
        out['lower_high'] = high.lt(high.shift(1))
        out['volume_contracts'] = data['volume'].lt(data['volume'].shift(1))
        out['valid'] = valid & valid.shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        x = indicators
        ready = x[['innovation', 'previous_innovation', 'drawdown', 'drawup']].notna().all(axis=1)
        ready &= x['valid'].fillna(False) & x['volume_contracts'].fillna(False)
        long_entry = (
            ready
            & x['previous_innovation'].le(-params.shock_threshold)
            & x['innovation'].gt(0)
            & x['higher_low'].fillna(False)
            & x['drawdown'].ge(params.shock_threshold)
        )
        short_entry = (
            ready
            & x['previous_innovation'].ge(params.shock_threshold)
            & x['innovation'].lt(0)
            & x['lower_high'].fillna(False)
            & x['drawup'].ge(params.shock_threshold)
        )
        events = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'innovation': [1.0, 0.0, -1.0, -1.0, 0.0, 1.0],
        'previous_innovation': [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
        'drawdown': 1.0, 'drawup': 1.0,
        'higher_low': True, 'lower_high': True,
        'volume_contracts': True, 'valid': True,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    assert result['size'].gt(0).all()
