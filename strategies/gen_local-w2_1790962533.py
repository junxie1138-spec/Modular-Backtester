from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    history_bars: int = 120
    neighbors: int = 7
    atr_bars: int = 14
    min_gap_atr: float = 0.25
    min_forecast: float = 0.001
    min_close_location: float = 0.65
    stop_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1790962533'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(1, int(params.history_bars)) + max(1, int(params.atr_bars)) + 7

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_bars = max(1, int(params.atr_bars))
        result['atr'] = true_range.rolling(atr_bars, min_periods=atr_bars).mean()
        result['gap_atr'] = (data['open'] - previous_close) / result['atr'].shift(1).replace(0.0, np.nan)
        result['gap_lag1'] = result['gap_atr'].shift(1)
        result['gap_lag2'] = result['gap_atr'].shift(2)
        session_range = (data['high'] - data['low']).replace(0.0, np.nan)
        result['close_location'] = (data['close'] - data['low']) / session_range
        result['body'] = data['close'] - data['open']

        history = max(1, int(params.history_bars))
        neighbors = min(history, max(1, int(params.neighbors)))
        shape = result[['gap_atr', 'gap_lag1', 'gap_lag2']]
        distances = []
        outcomes = []
        for lag in range(4, history + 4):
            distance = np.sqrt(((shape - shape.shift(lag)) ** 2).sum(axis=1, min_count=3)).to_numpy()
            # Historical entry is next open; exit is the open after two sessions.
            outcome = (data['open'].shift(lag - 3) / data['open'].shift(lag - 1).replace(0.0, np.nan) - 1.0).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(np.where(valid, distance, np.inf))
            outcomes.append(np.where(valid, outcome, 0.0))

        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argpartition(distance_matrix, neighbors - 1, axis=1)[:, :neighbors]
        selected_distances = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        result['analog_forecast'] = np.where(complete, selected_outcomes.mean(axis=1), np.nan)
        result['analog_distance'] = np.where(complete, selected_distances.mean(axis=1), np.nan)
        result['gap_agrees'] = ((result['gap_atr'] <= -max(0.0, params.min_gap_atr)) & (result['analog_forecast'] > params.min_forecast)).astype(int)
        result['candle_agrees'] = ((result['close_location'] >= params.min_close_location) & (result['body'] > 0.0)).astype(int)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entry = ((indicators['gap_agrees'] == 1) & (indicators['candle_agrees'] == 1)).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        highest_close = np.nan
        trailing_stop = np.nan
        stop_multiple = max(0.0, float(params.stop_atr))
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0.0
            if active:
                # The prior decision filled at this bar's open.
                held += 1
                if not valid:
                    active = False
                    continue
                highest_close = close if held == 1 else max(highest_close, close)
                candidate_stop = highest_close - stop_multiple * atr
                trailing_stop = candidate_stop if held == 1 else max(trailing_stop, candidate_stop)
                if close <= trailing_stop or held >= 2:
                    active = False
                else:
                    signals[i] = 1
            elif i >= warmup - 1 and valid and entry[i]:
                active = True
                held = 0
                highest_close = np.nan
                trailing_stop = np.nan
                signals[i] = 1

        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
