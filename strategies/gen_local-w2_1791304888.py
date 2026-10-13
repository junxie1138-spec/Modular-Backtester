from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    normalization_window: int = 63
    seasonal_window: int = 252
    seasonal_threshold: float = 0.75
    fractional_threshold: float = 0.25
    holding_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if self.fractional_window < 2 or self.normalization_window < 2:
            raise ValueError('fractional and normalization windows must exceed one')
        if self.seasonal_window < 100:
            raise ValueError('seasonal_window must be at least 100')
        if self.seasonal_threshold <= 0 or self.fractional_threshold <= 0:
            raise ValueError('thresholds must be positive')
        if self.holding_bars not in (3, 4, 5):
            raise ValueError('holding_bars must be three, four, or five')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791304888'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.fractional_window + params.normalization_window,
            params.seasonal_window + params.holding_bars,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        out['log_close'] = np.log(close)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, len(weights)):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: finite fractional memory; increase the window if truncation sensitivity matters.
        if len(data):
            values = np.convolve(out['log_close'].fillna(0).to_numpy(), weights, mode='full')[:len(data)]
            valid = out['log_close'].rolling(params.fractional_window).count().eq(params.fractional_window)
            out['fractional_close'] = pd.Series(values, index=data.index).where(valid)
        else:
            out['fractional_close'] = pd.Series(index=data.index, dtype=float)
        rolling = out['fractional_close'].rolling(params.normalization_window)
        out['fractional_mean'] = rolling.mean()
        out['fractional_std'] = rolling.std(ddof=0)
        out['fractional_z'] = (
            (out['fractional_close'] - out['fractional_mean'])
            / out['fractional_std'].replace(0, np.nan)
        )
        out['matured_return'] = close / close.shift(params.holding_bars) - 1.0
        out['origin_weekday'] = pd.Series(data.index.dayofweek, index=data.index).shift(params.holding_bars)
        out['weekday'] = data.index.dayofweek
        out['seasonal_mean'] = np.nan
        out['seasonal_score'] = np.nan
        out['seasonal_count'] = np.nan
        for weekday in range(5):
            observed = out['origin_weekday'].eq(weekday) & out['matured_return'].notna()
            samples = out['matured_return'].where(observed, 0.0)
            count = observed.astype(float).rolling(params.seasonal_window).sum()
            total = samples.rolling(params.seasonal_window).sum()
            squared_total = samples.pow(2).rolling(params.seasonal_window).sum()
            denominator = count.where(count >= 20)
            mean = total / denominator
            variance = (squared_total / denominator - mean.pow(2)).clip(lower=0)
            score = mean / (np.sqrt(variance) / np.sqrt(denominator)).replace(0, np.nan)
            out[f'weekday_{weekday}_count'] = count
            out[f'weekday_{weekday}_mean'] = mean
            out[f'weekday_{weekday}_score'] = score
            current = out['weekday'].eq(weekday)
            out.loc[current, 'seasonal_mean'] = mean.loc[current]
            out.loc[current, 'seasonal_score'] = score.loc[current]
            out.loc[current, 'seasonal_count'] = count.loc[current]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        seasonal = indicators['seasonal_score']
        fractional = indicators['fractional_z']
        long_entry = seasonal.ge(params.seasonal_threshold) & fractional.ge(params.fractional_threshold)
        short_entry = seasonal.le(-params.seasonal_threshold) & fractional.le(-params.fractional_threshold)
        candidate = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for i in range(len(data)):
            if side and i >= exit_bar:
                side = 0
                continue
            if not side and confirmed[i]:
                side = int(confirmed[i])
                exit_bar = i + params.holding_bars
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=13, freq='B')
    data = pd.DataFrame({'close': 100.0}, index=index)
    directions = [1.0] * 6 + [-1.0] * 7
    indicators = pd.DataFrame({'seasonal_score': directions, 'fractional_z': directions}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
