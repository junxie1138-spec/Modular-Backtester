from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    peak_bars: int = 63
    shape_bars: int = 5
    history_bars: int = 252
    neighbors: int = 12
    hold_bars: int = 8
    atr_bars: int = 14
    min_depth: float = 0.015
    forecast_floor: float = 0.004
    min_win_rate: float = 0.60
    max_distance: float = 0.008
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791289508'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(
            params.peak_bars + 2 * params.shape_bars
            + params.hold_bars + params.history_bars,
            params.atr_bars + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if min(params.peak_bars, params.shape_bars, params.history_bars,
               params.neighbors, params.hold_bars, params.atr_bars) < 1:
            raise ValueError('Lookbacks, neighbors, and holding period must be positive')
        if params.neighbors > params.history_bars:
            raise ValueError('Neighbors must not exceed history_bars')
        if params.stop_atr <= 0:
            raise ValueError('stop_atr must be positive')

        close = data['close'].astype(float)
        peak = close.rolling(params.peak_bars).max()
        depth = close.div(peak.where(peak > 0)).sub(1.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars).mean()

        out = pd.DataFrame(index=data.index)
        out['depth'] = depth
        out['atr'] = atr
        shape = pd.concat([
            depth.shift(offset).rename(f'depth_{offset}')
            for offset in range(params.shape_bars)
        ], axis=1)
        out[shape.columns] = shape
        current = shape.to_numpy(dtype=float)
        distances = np.full((len(data), params.history_bars), np.inf)
        outcomes = np.full_like(distances, np.nan)

        # Completed analog outcomes finish before the current shape begins.
        first_lag = params.hold_bars + params.shape_bars
        for column, lag in enumerate(range(
            first_lag, first_lag + params.history_bars
        )):
            historical = shape.shift(lag).to_numpy(dtype=float)
            distance = np.sqrt(np.mean((current - historical) ** 2, axis=1))
            historical_close = close.shift(lag)
            outcome = close.shift(lag - params.hold_bars).div(
                historical_close.where(historical_close > 0)
            ).sub(1.0).to_numpy(dtype=float)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[:, column] = np.where(valid, distance, np.inf)
            outcomes[:, column] = outcome

        chosen = np.argpartition(
            distances, params.neighbors - 1, axis=1
        )[:, :params.neighbors]
        selected_distance = np.take_along_axis(distances, chosen, axis=1)
        selected_outcome = np.take_along_axis(outcomes, chosen, axis=1)
        complete = np.isfinite(selected_distance).all(axis=1)
        safe_outcome = np.where(np.isfinite(selected_outcome), selected_outcome, 0.0)
        out['analog_forecast'] = np.where(
            complete, safe_outcome.mean(axis=1), np.nan
        )
        out['analog_win_rate'] = np.where(
            complete, (safe_outcome > 0).mean(axis=1), np.nan
        )
        safe_distance = np.where(np.isfinite(selected_distance), selected_distance, 0.0)
        out['analog_distance'] = np.where(
            complete, safe_distance.mean(axis=1), np.nan
        )
        out['eligible'] = (
            out['depth'].le(-params.min_depth)
            & out['depth'].gt(out['depth'].shift(1))
            & out['analog_forecast'].gt(params.forecast_floor)
            & out['analog_win_rate'].ge(params.min_win_rate)
            & out['analog_distance'].le(params.max_distance)
            & out['atr'].gt(0)
        ).fillna(False)
        out['confirmed'] = (
            out['eligible'] & out['eligible'].shift(1, fill_value=False)
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        live = False
        pending_entry = False
        fixed_stop = np.nan
        held_bars = 0

        for bar in range(len(data)):
            if pending_entry:
                # The prior decision fills at this bar's open. Freeze prior ATR.
                live = True
                pending_entry = False
                held_bars = 0
                fixed_stop = opening[bar] - params.stop_atr * atr[bar - 1]

            if live:
                held_bars += 1
                if close[bar] < fixed_stop or held_bars >= params.hold_bars:
                    live = False
                    signal[bar] = 0
                else:
                    signal[bar] = 1
            elif confirmed[bar] and np.isfinite(atr[bar]) and atr[bar] > 0:
                pending_entry = True
                signal[bar] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
