from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    block_bars: int = 4
    range_baseline: int = 20
    analog_window: int = 252
    neighbors: int = 7
    analog_distance: float = 0.25
    min_forecast: float = 0.006
    min_hit_rate: float = 0.57
    demand_bars: int = 5
    demand_threshold: float = 0.12
    atr_bars: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = 'gen_local-w2_1790991221'

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    def warmup_bars(self, params: RangeAnalogParams) -> int:
        return int(max(
            params.analog_window + params.range_baseline + 3 * params.block_bars + 1,
            params.atr_bars + 1,
            params.demand_bars,
            params.holding_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: RangeAnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        span = (data['high'] - data['low']).clip(lower=0)
        relative_range = span / previous_close.where(previous_close > 0)
        baseline = relative_range.rolling(params.range_baseline).mean()
        block_range = relative_range.rolling(params.block_bars).mean()
        for block in range(3):
            out[f'range_shape_{block}'] = (
                block_range.shift(block * params.block_bars)
                / baseline.where(baseline > 0)
            )

        true_range = pd.concat([
            span,
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_bars).mean()
        volume = data['volume'].clip(lower=0)
        candle_demand = ((close - data['open']) / span.where(span > 0)).clip(-1, 1).fillna(0)
        volume_total = volume.rolling(params.demand_bars).sum()
        out['demand'] = (
            (candle_demand * volume).rolling(params.demand_bars).sum()
            / volume_total.where(volume_total > 0)
        )

        features = out[[f'range_shape_{block}' for block in range(3)]]
        current = features.to_numpy(dtype=float)
        first_lag = params.holding_bars + 3 * params.block_bars
        lags = range(first_lag, params.analog_window + 1)
        distances = []
        outcomes = []
        for lag in lags:
            historical = features.shift(lag).to_numpy(dtype=float)
            distance = np.sqrt(np.mean((current - historical) ** 2, axis=1))
            outcome = (
                close.shift(lag - params.holding_bars)
                / close.shift(lag).where(close.shift(lag) > 0) - 1.0
            ).to_numpy(dtype=float)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(np.where(valid, distance, np.inf))
            outcomes.append(outcome)

        out['analog_forecast'] = np.nan
        out['analog_hit_rate'] = np.nan
        out['analog_count'] = 0
        if distances:
            distance_matrix = np.column_stack(distances)
            outcome_matrix = np.column_stack(outcomes)
            k = min(max(1, params.neighbors), distance_matrix.shape[1])
            nearest = np.argpartition(distance_matrix, k - 1, axis=1)[:, :k]
            selected_distance = np.take_along_axis(distance_matrix, nearest, axis=1)
            selected_outcome = np.take_along_axis(outcome_matrix, nearest, axis=1)
            accepted = (
                np.isfinite(selected_distance)
                & np.isfinite(selected_outcome)
                & (selected_distance <= params.analog_distance)
            )
            count = accepted.sum(axis=1)
            denominator = np.where(count > 0, count, np.nan)
            out['analog_forecast'] = np.where(accepted, selected_outcome, 0.0).sum(axis=1) / denominator
            out['analog_hit_rate'] = (accepted & (selected_outcome > 0)).sum(axis=1) / denominator
            out['analog_count'] = count

        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeAnalogParams,
    ) -> SignalFrame:
        analog_agrees = (
            (indicators['analog_count'] >= params.neighbors)
            & (indicators['analog_forecast'] >= params.min_forecast)
            & (indicators['analog_hit_rate'] >= params.min_hit_rate)
        )
        demand_agrees = indicators['demand'] >= params.demand_threshold
        entry = (analog_agrees & demand_agrees).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        fixed_stop = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]

            if active:
                stop_fired = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                time_exit = i - entry_bar + 1 >= params.holding_bars
                if stop_fired or time_exit:
                    active = False
                    continue
                raw[i] = 1
            elif (
                entry[i]
                and np.isfinite(closes[i])
                and closes[i] > 0
                and np.isfinite(atr[i])
                and atr[i] > 0
            ):
                raw[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
