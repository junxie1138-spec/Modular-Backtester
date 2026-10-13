from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    seasonal_window: int = 252
    range_percentile: float = 0.80
    min_events: int = 6
    profit_target: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791267885'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 2 * params.seasonal_window + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.seasonal_window < 100:
            raise ValueError('seasonal_window must be at least 100')
        if not 0.0 < params.range_percentile < 1.0:
            raise ValueError('range_percentile must lie between zero and one')
        if params.min_events < 1 or params.profit_target <= 0.0:
            raise ValueError('min_events and profit_target must be positive')
        if params.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions')
        out = pd.DataFrame(index=data.index)
        span = (data['high'] - data['low']).where(lambda x: x > 0.0)
        out['relative_range'] = span / data['close'].where(data['close'] > 0.0)
        out['expansion'] = (out['relative_range'] / out['relative_range'].shift(1)).replace([np.inf, -np.inf], np.nan)
        out['close_pressure'] = ((2.0 * data['close'] - data['high'] - data['low']) / span).clip(-1.0, 1.0)
        out['weekday'] = data.index.dayofweek
        out['range_threshold'] = np.nan
        window = params.seasonal_window
        for weekday in range(5):
            mask = out['weekday'].eq(weekday)
            threshold = out['expansion'].where(mask).rolling(window, min_periods=20).quantile(params.range_percentile).shift(1)
            out.loc[mask, 'range_threshold'] = threshold.loc[mask]
        out['range_event'] = out['expansion'].gt(out['range_threshold']) & out['range_threshold'].notna()
        out['intraday_return'] = (data['close'] / data['open'].where(data['open'] > 0.0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['follow_payoff'] = np.sign(out['close_pressure'].shift(1)) * out['intraday_return']
        out['seasonal_edge'] = np.nan
        out['event_count'] = 0.0
        previous_event = out['range_event'].shift(1, fill_value=False)
        previous_weekday = out['weekday'].shift(1)
        for weekday in range(5):
            historical_event = previous_event & previous_weekday.eq(weekday) & out['follow_payoff'].notna()
            count = historical_event.astype(float).rolling(window, min_periods=window).sum()
            total = out['follow_payoff'].where(historical_event, 0.0).rolling(window, min_periods=window).sum()
            edge = total / count.replace(0.0, np.nan)
            mask = out['weekday'].eq(weekday)
            out.loc[mask, 'seasonal_edge'] = edge.loc[mask]
            out.loc[mask, 'event_count'] = count.loc[mask]
        valid = out['range_event'] & out['event_count'].ge(params.min_events) & out['seasonal_edge'].notna() & out['close_pressure'].notna()
        out['entry_direction'] = np.sign(out['seasonal_edge'] * out['close_pressure']).where(valid, 0.0).fillna(0.0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        candidates = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_hit:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
                decisions[i] = position
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidates[i])
                entry_bar = -1
                decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_direction': [1, 0, -1, 0, 0, 0, 1, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 0, -1, -1, 0, 0, 1]
    assert result['size'].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(None, Params()) >= 506


if __name__ == '__main__':
    _self_check()
