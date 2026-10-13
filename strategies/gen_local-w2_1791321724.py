from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalPressureParams:
    seasonal_lookback: int = 504
    volume_window: int = 20
    volume_multiple: float = 1.3
    seasonal_threshold: float = 0.25
    min_body_return: float = 0.003
    min_events: int = 8
    holding_bars: int = 18

    def __post_init__(self):
        if self.seasonal_lookback < 1 or self.volume_window < 1:
            raise ValueError('Lookbacks must be positive.')
        if self.holding_bars < 1 or self.min_events < 1:
            raise ValueError('Holding bars and event count must be positive.')
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 0:
            raise ValueError('Volume multiple must be finite and positive.')
        if not np.isfinite(self.seasonal_threshold) or not 0 < self.seasonal_threshold <= 1:
            raise ValueError('Seasonal threshold must be in (0, 1].')
        if not np.isfinite(self.min_body_return) or self.min_body_return <= 0:
            raise ValueError('Minimum body return must be finite and positive.')


class GeneratedStrategy(BaseStrategy[SeasonalPressureParams]):
    strategy_id = 'gen_local-w2_1791321724'

    @classmethod
    def params_type(cls):
        return SeasonalPressureParams

    def warmup_bars(self, params: SeasonalPressureParams) -> int:
        return max(200, params.volume_window + 1, params.seasonal_lookback + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['volume_mean'] = data['volume'].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out['relative_volume'] = (
            data['volume'] / out['volume_mean'].where(out['volume_mean'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['body_return'] = (
            data['close'] / data['open'].where(data['open'] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out['confirmed_move'] = (
            out['relative_volume'].ge(params.volume_multiple)
            & out['body_return'].abs().ge(params.min_body_return)
        )
        out['pressure'] = out['body_return'].where(out['confirmed_move'], 0.0).fillna(0.0)
        out['calendar_month'] = data.index.month
        out['seasonal_score'] = np.nan
        out['seasonal_count'] = np.nan
        for month in range(1, 13):
            belongs = out['calendar_month'].eq(month)
            pressure = out['pressure'].where(belongs, 0.0)
            signed = pressure.rolling(
                params.seasonal_lookback, min_periods=params.seasonal_lookback
            ).sum().shift(1)
            absolute = pressure.abs().rolling(
                params.seasonal_lookback, min_periods=params.seasonal_lookback
            ).sum().shift(1)
            count = (belongs & out['confirmed_move']).astype(float).rolling(
                params.seasonal_lookback, min_periods=params.seasonal_lookback
            ).sum().shift(1)
            out['seasonal_score'] = out['seasonal_score'].where(
                ~belongs, signed / absolute.where(absolute > 0)
            )
            out['seasonal_count'] = out['seasonal_count'].where(~belongs, count)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalPressureParams,
    ) -> SignalFrame:
        ready = indicators['confirmed_move'] & indicators['seasonal_count'].ge(params.min_events)
        long_entry = (
            ready
            & indicators['body_return'].gt(0)
            & indicators['seasonal_score'].ge(params.seasonal_threshold)
            & data['close'].gt(indicators['ma200'])
        )
        short_entry = (
            ready
            & indicators['body_return'].lt(0)
            & indicators['seasonal_score'].le(-params.seasonal_threshold)
            & data['close'].lt(indicators['ma200'])
        )
        entries = np.select(
            [long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                # An exit consumes this decision bar; entry cannot cancel the flat signal.
            elif entry != 0:
                position = int(entry)
                exit_bar = bar + params.holding_bars
            signals[bar] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = SeasonalPressureParams(holding_bars=3)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=10, freq='B')
    expected = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    for direction in (1, -1):
        data = pd.DataFrame({'close': 100.0 + 10.0 * direction}, index=index)
        indicators = pd.DataFrame({
            'ma200': 100.0,
            'confirmed_move': True,
            'seasonal_count': 10.0,
            'body_return': 0.01 * direction,
            'seasonal_score': 0.5 * direction,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert np.array_equal(result['signal'].to_numpy(), expected * direction)
        assert result['size'].gt(0).all()
