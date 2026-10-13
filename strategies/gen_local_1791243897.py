from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    ma_window: int = 20
    analog_window: int = 126
    percentile_window: int = 126
    entry_percentile: float = 0.80
    neighbors: int = 5
    holding_bars: int = 4
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791243897'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(params.ma_window + params.analog_window + 2,
                       params.ma_window + params.percentile_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if (params.ma_window < 2 or params.percentile_window < 2
                or params.atr_window < 1 or not 3 <= params.holding_bars <= 5
                or params.neighbors < 1 or not 0 < params.entry_percentile < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError('Invalid strategy parameters')
        lags = list(range(params.holding_bars + 3, params.analog_window + 1,
                          params.holding_bars + 3))
        if len(lags) < params.neighbors:
            raise ValueError('Analog window must provide enough neighbors')
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['std'].replace(0.0, np.nan)
        out['z1'] = out['z'].shift(1)
        out['z2'] = out['z'].shift(2)
        out['cutoff'] = out['z'].abs().shift(1).rolling(
            params.percentile_window).quantile(params.entry_percentile)
        out['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - close.shift(1)).abs(),
            (data['low'] - close.shift(1)).abs()], axis=1).max(axis=1)
        out['atr'] = out['tr'].rolling(params.atr_window).mean()
        distances = []
        outcomes = []
        for lag in lags:
            distance = ((out['z'] - out['z'].shift(lag)) ** 2
                        + (out['z1'] - out['z1'].shift(lag)) ** 2
                        + (out['z2'] - out['z2'].shift(lag)) ** 2)
            outcome = close.shift(lag - params.holding_bars) / close.shift(lag) - 1.0
            out[f'distance_{lag}'] = distance
            out[f'outcome_{lag}'] = outcome
            distances.append(distance.to_numpy())
            outcomes.append(outcome.to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        valid = np.isfinite(distance_matrix) & np.isfinite(outcome_matrix)
        ranked = np.where(valid, distance_matrix, np.inf)
        nearest = np.argsort(ranked, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(ranked, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        out['analog_distance'] = np.where(complete, selected_distances.mean(axis=1), np.nan)
        out['forecast'] = np.where(complete, selected_outcomes.mean(axis=1), np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        eligible = (indicators['z'].abs().ge(indicators['cutoff'])
                    & indicators['forecast'].notna()
                    & indicators['atr'].gt(0))
        candidates = np.where(eligible,
                              np.sign(indicators['forecast'].fillna(0)), 0).astype(int)
        candidates[:self.warmup_bars(params)] = 0
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry = np.nan
        stop = np.nan
        armed = False
        age = 0
        for i in range(len(data)):
            if direction == 0:
                if candidates[i] and i + 1 < len(data):
                    direction = int(candidates[i])
                    entry = np.nan
                    armed = False
                    age = 0
                    decisions[i] = direction
                continue
            if not np.isfinite(entry):
                entry = opening[i]
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr[i - 1]):
                    direction = 0
                    continue
                # Signed-price stops only increase, including for short trades.
                stop = direction * entry - params.trail_atr * atr[i - 1]
            age += 1
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                direction = 0
                continue
            signed_close = direction * close[i]
            signed_entry = direction * entry
            # Close-confirmed exits fill at the following open.
            if signed_close <= stop or age >= params.holding_bars:
                direction = 0
                continue
            if signed_close - signed_entry >= entry * params.breakeven_pct:
                armed = True
            if armed:
                stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
                if signed_close <= stop:
                    direction = 0
                    continue
            decisions[i] = direction
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    t = np.arange(420, dtype=float)
    close = 100.0 + 0.02 * t + 4.0 * np.sin(t / 5.0)
    data = pd.DataFrame({'open': close + 0.1, 'high': close + 0.8,
                         'low': close - 0.8, 'close': close, 'volume': 1000000.0},
                        index=pd.date_range('2020-01-01', periods=len(t), freq='B'))
    strategy = GeneratedStrategy()
    params = ShapeParams()
    full = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    prefix = data.iloc[:350]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(full.iloc[:350], partial)
    assert full['signal'].isin([-1, 0, 1]).all()
    assert full['size'].gt(0).all()
    assert full['signal'].iloc[0] == 0
    assert full['signal'].ne(0).any()


if __name__ == '__main__':
    _self_check()
