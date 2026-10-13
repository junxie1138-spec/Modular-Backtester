from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    depth_window: int = 20
    atr_window: int = 14
    history_bars: int = 252
    hold_bars: int = 4
    match_distance: float = 0.8
    min_depth: float = 1.0
    atr_multiple: float = 1.7


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791326838'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.depth_window, params.atr_window) + params.history_bars + 5 + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.depth_window < 2 or params.atr_window < 2
                or params.hold_bars < 3 or params.hold_bars > 5
                or params.history_bars < params.hold_bars + 25
                or not np.isfinite(params.match_distance) or params.match_distance <= 0
                or not np.isfinite(params.min_depth) or params.min_depth <= 0
                or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0):
            raise ValueError('Invalid drawdown analog parameters')

        close = data['close']
        previous = close.shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1)
        true_range = ranges.max(axis=1).where(previous.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        peak = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        depth = ((peak - close) / atr.where(atr > 0)).replace([np.inf, -np.inf], np.nan)
        features = pd.concat([depth, depth.shift(2), depth.shift(4)], axis=1)
        features.columns = ['depth', 'depth_two_bars_ago', 'depth_four_bars_ago']

        distances = []
        outcomes = []
        # Historical five-bar shapes end before their outcomes mature and before the current shape begins.
        for lag in range(params.hold_bars + 5, params.history_bars + 1, 5):
            squared = (features - features.shift(lag)).pow(2)
            distance = np.sqrt(squared.sum(axis=1, min_count=3) / 3.0)
            base = close.shift(lag).where(close.shift(lag) > 0)
            outcome = close.shift(lag - params.hold_bars) / base - 1.0
            valid = distance.notna() & np.isfinite(outcome)
            distances.append(distance.where(valid, np.inf).to_numpy())
            outcomes.append(outcome.where(valid).to_numpy())

        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argsort(distance_matrix, axis=1)[:, :5]
        selected_distances = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1) & np.isfinite(selected_outcomes).all(axis=1)

        result = features.copy()
        result['atr'] = atr
        result['peak_close'] = peak
        result['analog_distance'] = np.where(complete, selected_distances.mean(axis=1), np.nan)
        result['analog_return'] = np.where(complete, selected_outcomes.mean(axis=1), np.nan)
        result['analog_win_rate'] = np.where(complete, (selected_outcomes > 0).mean(axis=1), np.nan)
        result['qualified'] = (
            (result['depth'] >= params.min_depth)
            & (result['analog_distance'] <= params.match_distance)
            & (result['analog_return'] > 0.002)
            & (result['analog_win_rate'] >= 0.6)
            & result['atr'].gt(0)
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        qualified = indicators['qualified'].fillna(False).astype(bool)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if active:
                age += 1
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    active = False
                else:
                    # Entry fills this bar when age is one: start tracking in-trade closes here.
                    high_water = close if age == 1 else max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    if close <= stop or age >= params.hold_bars:
                        active = False
            elif entries[i] and np.isfinite(close) and np.isfinite(atr) and atr > 0:
                active = True
                high_water = np.nan
                stop = -np.inf
                age = 0
            signals[i] = int(active)

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100, 100, 102, 104, 103, 100, 101, 101]}, index=index)
    indicators = pd.DataFrame({
        'qualified': [True, True, False, False, False, False, False, False],
        'atr': [1, 1, 1, 1, 10, 10, 1, 1],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, AnalogParams(hold_bars=5))
    # Confirms two-bar entry, one-bar execution delay, and a stop that cannot loosen as ATR rises.
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert result.data['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
