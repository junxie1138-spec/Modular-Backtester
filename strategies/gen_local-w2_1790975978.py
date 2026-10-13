from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityAnalogParams:
    graph_bars: int = 10
    history_bars: int = 252
    neighbors: int = 12
    holding_bars: int = 18
    max_distance: float = 0.20
    min_return: float = 0.01
    min_win_rate: float = 0.60


class GeneratedStrategy(BaseStrategy[VisibilityAnalogParams]):
    strategy_id = 'gen_local-w2_1790975978'

    @classmethod
    def params_type(cls):
        return VisibilityAnalogParams

    def warmup_bars(self, params):
        return int(params.holding_bars + params.history_bars + 2 * params.graph_bars)

    def indicators(self, data, params):
        if not (params.graph_bars >= 3 and params.holding_bars >= 1
                and 1 <= params.neighbors <= params.history_bars
                and 0 <= params.max_distance <= 1
                and 0 <= params.min_win_rate <= 1
                and np.isfinite(params.min_return)):
            raise ValueError('Invalid visibility analog parameters')
        n = len(data)
        w = params.graph_bars
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        log_price = np.log(close)
        paths = np.column_stack([
            log_price.shift(w - 1 - j).to_numpy() for j in range(w)
        ])
        valid = np.isfinite(paths).all(axis=1)
        result = pd.DataFrame(index=data.index)
        edges = []
        for left in range(w - 2):
            for right in range(left + 2, w):
                # Natural visibility: every intervening vertex lies below
                # the straight line connecting the two endpoints.
                visible = valid.copy()
                for middle in range(left + 1, right):
                    fraction = (middle - left) / (right - left)
                    boundary = paths[:, left] + fraction * (
                        paths[:, right] - paths[:, left]
                    )
                    visible &= paths[:, middle] < boundary
                edge = np.where(valid, visible.astype(float), np.nan)
                result[f'edge_{left}_{right}'] = edge
                edges.append(edge)
        topology = np.column_stack(edges)
        distances = np.full((n, params.history_bars), np.inf)
        outcomes = np.full((n, params.history_bars), np.nan)
        prices = close.to_numpy()
        first_lag = params.holding_bars + w
        # ponytail: exhaustive O(rows * history * edges) matching;
        # use indexed binary signatures if much larger histories are needed.
        for column in range(params.history_bars):
            lag = first_lag + column
            if lag >= n:
                break
            historical_return = (
                prices[params.holding_bars:n - lag + params.holding_bars]
                / prices[:n - lag] - 1.0
            )
            distance = np.mean(topology[lag:] != topology[:-lag], axis=1)
            usable = (valid[lag:] & valid[:-lag]
                      & np.isfinite(historical_return)
                      & (distance <= params.max_distance))
            distances[lag:, column] = np.where(usable, distance, np.inf)
            outcomes[lag:, column] = historical_return
        # A candidate's outcome finishes at least one graph window ago,
        # keeping its historical path and outcome separate from today's graph.
        nearest = np.argsort(distances, axis=1, kind='stable')[:, :params.neighbors]
        selected_distance = np.take_along_axis(distances, nearest, axis=1)
        selected_return = np.take_along_axis(outcomes, nearest, axis=1)
        usable = np.isfinite(selected_distance) & np.isfinite(selected_return)
        count = usable.sum(axis=1)
        denominator = np.maximum(count, 1)
        forecast = np.where(usable, selected_return, 0.0).sum(axis=1) / denominator
        win_rate = (usable & (selected_return > 0.0)).sum(axis=1) / denominator
        ready = (count == params.neighbors) & valid
        ready &= np.arange(n) >= self.warmup_bars(params) - 1
        result['analog_count'] = count
        result['forecast_return'] = np.where(ready, forecast, np.nan)
        result['analog_win_rate'] = np.where(ready, win_rate, np.nan)
        return result

    def generate_signals(self, data, indicators, ctx, params):
        eligible = (
            indicators['forecast_return'].ge(params.min_return)
            & indicators['analog_win_rate'].ge(params.min_win_rate)
            & indicators['analog_count'].ge(params.neighbors)
        ).fillna(False)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    # Exit always wins on this bar, even if entry qualifies.
                    exit_bar = -1
            elif confirmed[bar]:
                signal[bar] = 1
                exit_bar = bar + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = VisibilityAnalogParams(graph_bars=5, history_bars=32,
                                   neighbors=4, holding_bars=18)
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 160)))
    data = pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close,
                         'volume': 1000000.0},
                        index=pd.date_range('2020-01-01', periods=160, freq='B'))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:120], params)
    pd.testing.assert_frame_equal(full.iloc[:120], prefix)
    ctx = StrategyContext(symbol='SPY', timeframe='1d',
                          warmup_bars=strategy.warmup_bars(params))
    forced = pd.DataFrame({'forecast_return': 0.10,
                           'analog_win_rate': 1.0,
                           'analog_count': params.neighbors}, index=data.index)
    forced.loc[data.index[:10], 'forecast_return'] = np.nan
    # Eligible at bars 10 and 11; entry fills at bar 12.
    signals = strategy.generate_signals(data, forced, ctx, params).data['signal']
    assert signals.iloc[:12].eq(0).all()
    assert signals.iloc[12:30].eq(1).all()
    assert signals.iloc[30] == 0
    assert signals.isin([0, 1]).all()
    isolated = forced.copy()
    isolated['forecast_return'] = np.nan
    isolated.iloc[10, isolated.columns.get_loc('forecast_return')] = 0.10
    assert strategy.generate_signals(data, isolated, ctx, params).data['signal'].eq(0).all()
    altered = forced.copy()
    altered.iloc[12:, altered.columns.get_loc('forecast_return')] = np.nan
    held = strategy.generate_signals(data, altered, ctx, params).data['signal']
    assert held.iloc[12:30].eq(1).all() and held.iloc[30:].eq(0).all()
