from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    season_window: int = 252
    threshold_window: int = 126
    entry_percentile: float = 0.75
    hold_bars: int = 4

    def __post_init__(self):
        if self.season_window < 10 or self.threshold_window < 2:
            raise ValueError('Lookback windows are too short')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must lie between zero and one')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five')


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = 'gen_local-w2_1790982043'

    @classmethod
    def params_type(cls):
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return int(params.season_window + params.threshold_window + 2)

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        weekday = data.index.dayofweek.to_numpy()
        out['weekday'] = weekday
        phase_columns = []
        for day in range(5):
            mask = pd.Series(weekday == day, index=data.index)
            total_name = f'weekday_total_{day}'
            count_name = f'weekday_count_{day}'
            mean_name = f'weekday_mean_{day}'
            observations = mask & out['return'].notna()
            out[total_name] = out['return'].where(observations, 0.0).shift(1).rolling(params.season_window, min_periods=params.season_window).sum()
            out[count_name] = observations.astype(float).shift(1).rolling(params.season_window, min_periods=params.season_window).sum()
            out[mean_name] = out[total_name].div(out[count_name].replace(0.0, np.nan))
            phase_columns.append(mean_name)
        phase_values = out[phase_columns].to_numpy()
        rows = np.arange(len(data))
        forecast = pd.Series(0.0, index=data.index)
        for step in range(1, params.hold_bars + 1):
            name = f'forecast_step_{step}'
            out[name] = phase_values[rows, (weekday + step) % 5]
            forecast = forecast + out[name]
        out['forecast'] = forecast.replace([np.inf, -np.inf], np.nan)
        out['strength'] = out['forecast'].abs()
        out['threshold'] = out['strength'].shift(1).rolling(params.threshold_window, min_periods=params.threshold_window).quantile(params.entry_percentile)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalParams) -> SignalFrame:
        valid = indicators[['forecast', 'strength', 'threshold']].notna().all(axis=1)
        eligible = valid & indicators['strength'].gt(indicators['threshold'])
        candidates = np.where(eligible, np.sign(indicators['forecast'].fillna(0.0)), 0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position:
                if i == exit_bar:
                    position = 0
            elif candidate:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.bdate_range('2020-01-01', periods=14)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    forecast = np.where(np.arange(len(index)) % 2 == 0, 1.0, -1.0)
    indicators = pd.DataFrame({'forecast': forecast, 'strength': 1.0, 'threshold': 0.5}, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 1, 1, 1]
    assert result['size'].gt(0.0).all()
    warm = strategy.indicators(data, params)
    assert strategy.generate_signals(data, warm, ctx, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
