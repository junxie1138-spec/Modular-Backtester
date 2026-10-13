from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history_bars: int = 504
    horizon_bars: int = 8
    atr_period: int = 14
    shape_bars: int = 5
    min_analogs: int = 8
    shape_bandwidth: float = 1.0
    min_win_rate: float = 0.58
    min_edge_atr: float = 0.25
    trailing_atr: float = 2.5
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1790990929'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(1, params.history_bars) + max(1, params.horizon_bars) + max(max(1, params.atr_period) + 1, max(1, params.shape_bars) + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        delta = close.diff()
        up = delta.gt(0)
        down = delta.lt(0)
        out['up_streak'] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out['down_streak'] = down.astype(int).groupby((~down).cumsum()).cumsum()
        out['preceding_down'] = out['down_streak'].shift(1).where(out['up_streak'].eq(1)).ffill().fillna(0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_period = max(1, int(params.atr_period))
        out['atr'] = out['true_range'].rolling(atr_period, min_periods=atr_period).mean()
        valid_atr = out['atr'].where(out['atr'].gt(0))
        shape_bars = max(1, int(params.shape_bars))
        out['shape'] = (close - close.shift(shape_bars)) / valid_atr
        out['eligible'] = out['up_streak'].between(1, 3) & out['preceding_down'].between(2, 6)
        weight_sum = pd.Series(0.0, index=data.index)
        squared_weight_sum = pd.Series(0.0, index=data.index)
        payoff_sum = pd.Series(0.0, index=data.index)
        win_sum = pd.Series(0.0, index=data.index)
        count = pd.Series(0, index=data.index, dtype=int)
        horizon = max(1, int(params.horizon_bars))
        history = max(1, int(params.history_bars))
        bandwidth = max(0.01, float(params.shape_bandwidth))
        # Each historical outcome ends at least one bar before the current close.
        for lag in range(horizon + 1, horizon + history + 1):
            matched = (
                out['eligible']
                & out['eligible'].shift(lag).fillna(False).astype(bool)
                & out['up_streak'].eq(out['up_streak'].shift(lag))
                & out['preceding_down'].eq(out['preceding_down'].shift(lag))
            )
            distance = (out['shape'] - out['shape'].shift(lag)).abs()
            payoff = (close.shift(lag - horizon) - close.shift(lag)) / valid_atr.shift(lag)
            usable = matched & distance.le(bandwidth) & payoff.notna() & np.isfinite(payoff)
            weight = (1.0 - distance / bandwidth).clip(lower=0.0).where(usable, 0.0).fillna(0.0)
            weight_sum += weight
            squared_weight_sum += weight.pow(2)
            payoff_sum += weight * payoff.fillna(0.0)
            win_sum += weight * payoff.gt(0).astype(float)
            count += weight.gt(0).astype(int)
        denominator = weight_sum.where(weight_sum.gt(0))
        out['analog_count'] = count
        out['weight_sum'] = weight_sum
        out['squared_weight_sum'] = squared_weight_sum
        out['payoff_sum'] = payoff_sum
        out['win_sum'] = win_sum
        out['edge_atr'] = (payoff_sum / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['win_rate'] = (win_sum / denominator).fillna(0.0)
        out['effective_analogs'] = (weight_sum.pow(2) / squared_weight_sum.where(squared_weight_sum.gt(0))).fillna(0.0)
        required = max(1, int(params.min_analogs))
        out['support'] = (out['effective_analogs'] / (2.0 * required)).clip(0.0, 1.0)
        out['strength'] = (
            out['edge_atr'].clip(0.0, 2.0) / 2.0
            * ((out['win_rate'] - 0.5) * 2.0).clip(0.0, 1.0)
            * out['support']
        ).fillna(0.0)
        out['ready'] = pd.Series(np.arange(len(data)), index=data.index).ge(GeneratedStrategy.warmup_bars(params) - 1)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        required = max(1, int(params.min_analogs))
        entries = (
            indicators['ready']
            & indicators['eligible']
            & indicators['analog_count'].ge(required)
            & indicators['effective_analogs'].ge(required)
            & indicators['win_rate'].ge(params.min_win_rate)
            & indicators['edge_atr'].ge(params.min_edge_atr)
            & indicators['strength'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        minimum_size = float(np.clip(params.min_size, 0.001, 1.0))
        maximum_size = float(np.clip(params.max_size, minimum_size, 1.0))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum_size, dtype=float)
        active = False
        high_water = 0.0
        stop = 0.0
        age = 0
        horizon = max(1, int(params.horizon_bars))
        multiple = max(0.01, float(params.trailing_atr))
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                continue
            if active:
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - multiple * atr[i])
                if close[i] <= stop or age >= horizon:
                    active = False
                    continue
            elif entries[i]:
                active = True
                age = 0
                high_water = close[i]
                stop = high_water - multiple * atr[i]
            if active:
                signals[i] = 1
                sizes[i] = minimum_size + (maximum_size - minimum_size) * float(np.clip(strength[i], 0.0, 1.0))
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
