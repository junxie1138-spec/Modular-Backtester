from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_period: int = 3
    volatility_window: int = 20
    history_bars: int = 90
    shape_scale: float = 0.75
    forecast_threshold: float = 0.001
    minimum_matches: float = 8.0
    hold_bars: int = 2

    def __post_init__(self):
        if self.roc_period < 1 or self.volatility_window < 2:
            raise ValueError('Invalid indicator lookback')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if self.history_bars < self.hold_bars + 3:
            raise ValueError('Insufficient analog history')
        if not all(np.isfinite(x) and x > 0 for x in (
            self.shape_scale, self.forecast_threshold, self.minimum_matches
        )):
            raise ValueError('Thresholds must be finite and positive')


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791259440'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return params.history_bars + max(
            params.volatility_window + 1, params.roc_period + 2
        ) + 2

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        opening = data['open'].astype(float).where(data['open'] > 0)
        out['daily_return'] = close.pct_change(fill_method=None)
        out['roc'] = close.pct_change(params.roc_period, fill_method=None)
        out['volatility'] = out['daily_return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0).replace(0.0, np.nan)
        out['acceleration'] = out['roc'].diff() / out['volatility']
        out['acceleration_1'] = out['acceleration'].shift(1)
        out['acceleration_2'] = out['acceleration'].shift(2)
        shape = out[['acceleration', 'acceleration_1', 'acceleration_2']].replace(
            [np.inf, -np.inf], np.nan
        )
        weighted_return = pd.Series(0.0, index=data.index)
        weight_sum = pd.Series(0.0, index=data.index)
        weight_squared = pd.Series(0.0, index=data.index)
        for lag in range(params.hold_bars + 3, params.history_bars + 1):
            distance = (shape - shape.shift(lag)).pow(2).sum(
                axis=1, min_count=3
            ) / 3.0
            # Analog entry and exit use the same next-open convention as fills.
            outcome = opening.shift(lag - params.hold_bars - 1) / opening.shift(lag - 1) - 1.0
            valid = distance.notna() & np.isfinite(outcome)
            weight = np.exp(-distance / (2.0 * params.shape_scale ** 2)).where(valid, 0.0)
            weighted_return += weight * outcome.where(valid, 0.0)
            weight_sum += weight
            weight_squared += weight.pow(2)
        out['weight_sum'] = weight_sum
        out['effective_matches'] = weight_sum.pow(2) / weight_squared.replace(0.0, np.nan)
        out['forecast'] = weighted_return / weight_sum.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        ready = (
            indicators['forecast'].notna()
            & indicators['effective_matches'].ge(params.minimum_matches)
            & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        )
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[ready & indicators['forecast'].gt(params.forecast_threshold)] = 1
        candidate.loc[ready & indicators['forecast'].lt(-params.forecast_threshold)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar, direction in enumerate(confirmed):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            elif direction != 0:
                position = int(direction)
                entry_bar = bar
            signal[bar] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = AnalogParams()
    strategy = object.__new__(GeneratedStrategy)
    warmup = strategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=warmup + 8)
    data = pd.DataFrame(index=index)
    forecast = np.zeros(len(index))
    forecast[warmup:] = [0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    indicators = pd.DataFrame({'forecast': forecast, 'effective_matches': 10.0}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[warmup:].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
