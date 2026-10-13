from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 63
    phase_bars: int = 5
    analog_history: int = 126
    forecast_window: int = 63
    analog_fraction: float = 0.15
    tail_fraction: float = 0.20
    holding_bars: int = 7


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791331068'

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        if min(params.rank_window, params.phase_bars, params.forecast_window) < 2:
            raise ValueError('Rolling windows must be at least two bars')
        if params.analog_history < 10 or not 5 <= params.holding_bars <= 10:
            raise ValueError('Require at least ten analogs and a five-to-ten-bar hold')
        if not 0 < params.analog_fraction < 1 or not 0 < params.tail_fraction < 0.5:
            raise ValueError('Invalid percentile fractions')
        return int(params.rank_window + 1 + 4 * params.phase_bars + params.holding_bars + params.analog_history + params.forecast_window)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        primitives = {
            'return': close.pct_change(fill_method=None),
            'volume': data['volume'].where(data['volume'] >= 0),
            'range': (data['high'] - data['low']).where(data['high'] >= data['low']) / close,
        }
        features = []
        for name, series in primitives.items():
            rank_name = name + '_rank'
            out[rank_name] = series.replace([np.inf, -np.inf], np.nan).rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
            late_name = name + '_late'
            early_name = name + '_early'
            out[late_name] = out[rank_name].rolling(params.phase_bars, min_periods=params.phase_bars).mean()
            out[early_name] = out[late_name].shift(params.phase_bars)
            features.extend([early_name, late_name])
        shape = out[features]
        distances = []
        outcomes = []
        first_lag = params.holding_bars + 2 * params.phase_bars
        # ponytail: O(bars * analog_history) memory; batch candidates for much longer histories.
        for lag in range(first_lag, first_lag + params.analog_history):
            delta = shape - shape.shift(lag)
            distances.append(delta.pow(2).sum(axis=1, min_count=len(features)).to_numpy())
            outcome = close.shift(lag - params.holding_bars) / close.shift(lag) - 1.0
            outcomes.append(outcome.to_numpy())
        distance = np.column_stack(distances)
        outcome = np.column_stack(outcomes)
        valid = np.isfinite(distance).all(axis=1) & np.isfinite(outcome).all(axis=1)
        safe_distance = np.where(valid[:, None], distance, 0.0)
        cutoff = np.quantile(safe_distance, params.analog_fraction, axis=1)
        selected = (safe_distance <= cutoff[:, None]) & valid[:, None]
        count = selected.sum(axis=1)
        total = np.where(selected, outcome, 0.0).sum(axis=1)
        forecast = np.full(len(data), np.nan, dtype=float)
        np.divide(total, count, out=forecast, where=count > 0)
        out['distance_cutoff'] = np.where(valid, cutoff, np.nan)
        out['analog_count'] = np.where(valid, count, np.nan)
        out['forecast'] = forecast
        out['forecast_rank'] = out['forecast'].rolling(params.forecast_window, min_periods=params.forecast_window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        rank = indicators['forecast_rank']
        forecast = indicators['forecast']
        eligible = np.isfinite(rank) & np.isfinite(forecast)
        long_entry = eligible & (rank >= 1.0 - params.tail_fraction) & (forecast > 0)
        short_entry = eligible & (rank <= params.tail_fraction) & (forecast < 0)
        entries = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = ShapeParams()
    index = pd.date_range('2020-01-01', periods=20)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'forecast_rank': [0.95] + [0.05] * 19, 'forecast': [0.01] + [-0.01] * 19}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    signal = result['signal'].to_numpy()
    assert signal[0] == 0
    assert (signal[1:8] == 1).all()
    assert signal[8] == 0
    assert (signal[9:16] == -1).all()
    assert signal[16] == 0
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
