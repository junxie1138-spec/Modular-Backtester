from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAccelerationParams:
    roc_bars: int = 3
    seasonal_window: int = 126
    noise_window: int = 40
    shock_threshold: float = 0.75
    profit_target: float = 0.012
    hold_bars: int = 4

    def __post_init__(self):
        if self.roc_bars < 1 or self.seasonal_window < 40 or self.noise_window < 2:
            raise ValueError('Invalid indicator lookbacks')
        if not np.isfinite(self.shock_threshold) or self.shock_threshold <= 0:
            raise ValueError('shock_threshold must be finite and positive')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if self.hold_bars < 3 or self.hold_bars > 5:
            raise ValueError('hold_bars must be between three and five')


class GeneratedStrategy(BaseStrategy[SeasonalAccelerationParams]):
    strategy_id = 'gen_local_1791258014'

    @classmethod
    def params_type(cls):
        return SeasonalAccelerationParams

    def warmup_bars(self, params: SeasonalAccelerationParams) -> int:
        return int(max(params.seasonal_window + params.roc_bars + 2,
                       params.noise_window + params.roc_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SeasonalAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff().replace([np.inf, -np.inf], np.nan)
        out['noise'] = out['acceleration'].rolling(params.noise_window).std(ddof=0)
        out['seasonal_acceleration'] = np.nan
        out['seasonal_count'] = 0.0
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        for day in range(5):
            samples = out['acceleration'].where(weekday.eq(day)).shift(1)
            count = samples.notna().astype(float).rolling(params.seasonal_window).sum()
            total = samples.fillna(0.0).rolling(params.seasonal_window).sum()
            mean = total.div(count.where(count >= 8))
            mask = weekday.eq(day)
            out.loc[mask, 'seasonal_acceleration'] = mean.loc[mask]
            out.loc[mask, 'seasonal_count'] = count.loc[mask]
        out['shock'] = (out['acceleration'] - out['seasonal_acceleration']).div(
            out['noise'].where(out['noise'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        ready = out[['acceleration', 'seasonal_acceleration', 'noise', 'shock']].notna().all(axis=1)
        long_entry = ready & out['seasonal_acceleration'].gt(0) & out['acceleration'].lt(0) & out['shock'].le(-params.shock_threshold)
        short_entry = ready & out['seasonal_acceleration'].lt(0) & out['acceleration'].gt(0) & out['shock'].ge(params.shock_threshold)
        out['candidate'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalAccelerationParams) -> SignalFrame:
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        decisions[i] = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                # Close-observed targets request an exit at the following open.
                if (np.isfinite(gain) and gain >= params.profit_target) or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    entry_bar = -1
            elif confirmed[i]:
                position = int(confirmed[i])
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = SeasonalAccelerationParams()
    index = pd.bdate_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    test_indicators = pd.DataFrame({'candidate': [1, 1, 0, 0, 0, 0, 0, 0]}, index=index)
    def signals():
        return strategy.generate_signals(data, test_indicators, None, params).data['signal'].tolist()
    assert signals() == [0, 0, 1, 1, 1, 1, 0, 0]
    data.loc[index[3], 'close'] = 102.0
    assert signals() == [0, 0, 1, 1, 0, 0, 0, 0]
    test_indicators['candidate'] *= -1
    data.loc[index[3], 'close'] = 98.0
    assert signals() == [0, 0, -1, -1, 0, 0, 0, 0]
    test_indicators['candidate'] = [1, 0, 1, 0, 1, 0, 1, 0]
    assert signals() == [0] * 8
