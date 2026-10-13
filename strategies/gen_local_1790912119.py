from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSeasonParams:
    range_lookback: int = 20
    season_lookback: int = 252
    hold_bars: int = 4
    entry_threshold: float = 0.18
    min_size: float = 0.25
    max_size: float = 0.95
    full_size_score: float = 0.75


class GeneratedStrategy(BaseStrategy[RangeSeasonParams]):
    strategy_id = 'gen_local_1790912119'

    @classmethod
    def params_type(cls) -> type[RangeSeasonParams]:
        return RangeSeasonParams

    @staticmethod
    def warmup_bars(params: RangeSeasonParams) -> int:
        if params.range_lookback < 2 or params.season_lookback < 60:
            raise ValueError('Lookbacks must be at least 2 and 60 bars.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between 3 and 5.')
        if not 0 < params.entry_threshold < params.full_size_score:
            raise ValueError('Require 0 < entry_threshold < full_size_score.')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Require 0 < min_size <= max_size <= 1.')
        return int(params.range_lookback + params.season_lookback + params.hold_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeSeasonParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['weekday'] = data.index.dayofweek
        out['range'] = (data['high'] - data['low']).where(lambda x: x > 0)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['range_fraction'] = out['range'] / data['close'].where(data['close'] > 0)
        out['range_baseline'] = out['range_fraction'].shift(1).rolling(params.range_lookback, min_periods=params.range_lookback).mean()
        out['expansion'] = (out['range_fraction'] / out['range_baseline'] - 1.0).clip(-1.0, 2.0)
        out['migration'] = (out['midpoint'].diff() / out['range'].shift(1)).clip(-2.0, 2.0)
        out['shock'] = out['expansion'] * out['migration']
        # At t, the sample entered at open[t-h] and exited at open[t].
        # Its decision feature belongs to t-h-1, so every label is completed.
        lag = params.hold_bars + 1
        out['matured_shock'] = out['shock'].shift(lag)
        out['origin_weekday'] = out['weekday'].shift(lag)
        out['realized_return'] = data['open'] / data['open'].shift(params.hold_bars).where(data['open'].shift(params.hold_bars) > 0) - 1.0
        out['score'] = 0.0
        valid = np.isfinite(out['matured_shock']) & np.isfinite(out['realized_return'])
        for weekday in range(5):
            mask = valid & out['origin_weekday'].eq(weekday)
            prefix = 'weekday_' + str(weekday) + '_'
            out[prefix + 'count'] = mask.astype(float).rolling(params.season_lookback, min_periods=params.season_lookback).sum()
            out[prefix + 'xx'] = out['matured_shock'].pow(2).where(mask, 0.0).rolling(params.season_lookback, min_periods=params.season_lookback).sum()
            out[prefix + 'xy'] = (out['matured_shock'] * out['realized_return']).where(mask, 0.0).rolling(params.season_lookback, min_periods=params.season_lookback).sum()
            out[prefix + 'yy'] = out['realized_return'].pow(2).where(mask, 0.0).rolling(params.season_lookback, min_periods=params.season_lookback).sum()
            out[prefix + 'slope'] = out[prefix + 'xy'] / out[prefix + 'xx'].where(out[prefix + 'xx'] > 1e-12)
            out[prefix + 'risk'] = np.sqrt(out[prefix + 'yy'] / out[prefix + 'count'].where(out[prefix + 'count'] >= 10))
            out[prefix + 'forecast_score'] = out['shock'] * out[prefix + 'slope'] / out[prefix + 'risk'].where(out[prefix + 'risk'] > 1e-12)
            today = out['weekday'].eq(weekday)
            out.loc[today, 'score'] = out.loc[today, prefix + 'forecast_score']
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeSeasonParams) -> SignalFrame:
        warmup = GeneratedStrategy.warmup_bars(params)
        scores = indicators['score'].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if direction != 0:
                if i == exit_bar:
                    direction = 0
                    continue
            elif i >= warmup - 1 and abs(scores[i]) >= params.entry_threshold:
                direction = 1 if scores[i] > 0 else -1
                exit_bar = i + params.hold_bars
                strength = np.clip((abs(scores[i]) - params.entry_threshold) / (params.full_size_score - params.entry_threshold), 0.0, 1.0)
                entry_size = float(params.min_size + strength * (params.max_size - params.min_size))
            if direction != 0:
                signals[i] = direction
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RangeSeasonParams()
    decision = GeneratedStrategy.warmup_bars(params) - 1
    index = pd.date_range('2020-01-01', periods=decision + 16, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'score': np.zeros(len(index))}, index=index)
    indicators.iloc[decision:, 0] = -1.0
    indicators.iloc[decision, 0] = 1.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    entry = decision + 1
    exit_index = entry + params.hold_bars
    assert result['signal'].iloc[:entry].eq(0).all()
    assert result['signal'].iloc[entry:exit_index].eq(1).all()
    assert result['signal'].iloc[exit_index] == 0
    assert result['signal'].iloc[exit_index + 1] == -1
    assert result['size'].iloc[entry:exit_index].eq(params.max_size).all()
    assert result['size'].gt(0).all()
    assert set(result['signal'].unique()) <= {-1, 0, 1}


if __name__ == '__main__':
    _self_check()
