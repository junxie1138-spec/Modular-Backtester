from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    complexity_window: int = 12
    percentile_window: int = 126
    analog_window: int = 252
    complexity_quantile: float = 0.40
    forecast_quantile: float = 0.70
    min_analogs: int = 3
    atr_window: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1790999615'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.atr_window + 1,
                       params.complexity_window + 2 * params.percentile_window
                       + params.analog_window + 8))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag1'] = out['return'].shift(1)
        out['return_lag2'] = out['return'].shift(2)
        a, b, c = out['return_lag2'], out['return_lag1'], out['return']
        valid = a.notna() & b.notna() & c.notna()
        out['pattern'] = (4 * (a > b).astype(int)
                          + 2 * (a > c).astype(int)
                          + (b > c).astype(int)).astype(float).where(valid)
        codes = (0, 1, 3, 4, 6, 7)
        entropy_terms = []
        for code in codes:
            probability = out['pattern'].eq(code).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
            out[f'probability_{code}'] = probability
            entropy_terms.append(-probability * np.log(probability.where(probability > 0, 1.0)))
        out['complexity'] = pd.concat(entropy_terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        out['complexity_threshold'] = out['complexity'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.complexity_quantile)
        out['ordered_episode'] = (out['complexity'] <= out['complexity_threshold']).astype(float).where(
            out['complexity'].notna() & out['complexity_threshold'].notna()
        )
        out['signature'] = 8 * out['pattern'].shift(1) + out['pattern']
        # At t, the sample starts at t-3, enters at open t-2, and ends at close t-1.
        out['mature_signature'] = out['signature'].shift(3)
        out['mature_episode'] = out['ordered_episode'].shift(3)
        out['mature_outcome'] = (data['close'].shift(1) / data['open'].shift(2).replace(0, np.nan) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        analog_count = pd.Series(np.nan, index=data.index, dtype=float)
        for previous in codes:
            for current in codes:
                signature = 8 * previous + current
                match = (out['mature_signature'].eq(signature)
                         & out['mature_episode'].eq(1.0)
                         & out['mature_outcome'].notna())
                count = match.astype(float).rolling(
                    params.analog_window, min_periods=params.analog_window
                ).sum()
                total = out['mature_outcome'].where(match, 0.0).rolling(
                    params.analog_window, min_periods=params.analog_window
                ).sum()
                out[f'analog_count_{signature}'] = count
                out[f'analog_sum_{signature}'] = total
                estimate = (total / count.replace(0, np.nan)).where(count >= params.min_analogs)
                selected = out['signature'].eq(signature)
                forecast = forecast.where(~selected, estimate)
                analog_count = analog_count.where(~selected, count)
        out['analog_count'] = analog_count
        out['forecast'] = forecast
        out['forecast_magnitude'] = forecast.abs()
        out['forecast_threshold'] = out['forecast_magnitude'].shift(1).rolling(
            params.percentile_window, min_periods=min(20, params.percentile_window)
        ).quantile(params.forecast_quantile)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        eligible = (indicators['ordered_episode'].eq(1.0)
                    & indicators['forecast_magnitude'].gt(indicators['forecast_threshold'])
                    & indicators['atr'].gt(0))
        candidates = np.sign(indicators['forecast'].where(eligible, 0.0).fillna(0.0)).astype(int).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        frozen_atr = np.nan
        stop_price = np.nan
        age = 0
        holding_limit = max(1, min(2, int(params.hold_bars)))
        ready_at = self.warmup_bars(params) - 1
        for i in range(len(data)):
            # A decision made at the preceding close fills at this session's open.
            if position == 0 and pending != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(frozen_atr):
                    position = pending
                    stop_price = opens[i] - position * params.stop_atr * frozen_atr
                    age = 0
                pending = 0
            if position != 0:
                age += 1
                stopped = (np.isfinite(closes[i])
                           and position * (closes[i] - stop_price) < 0)
                if stopped or age >= holding_limit:
                    # Zero at this close becomes the exit instruction on the next bar.
                    decisions[i] = 0
                    position = 0
                    continue
                decisions[i] = position
            elif i >= ready_at and candidates[i] != 0:
                if np.isfinite(atr[i]) and atr[i] > 0:
                    pending = int(candidates[i])
                    frozen_atr = atr[i]
                    decisions[i] = pending
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
