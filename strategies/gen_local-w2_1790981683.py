from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    history_bars: int = 504
    max_streak: int = 5
    prior_split: int = 2
    min_analogs: int = 6
    min_edge: float = 0.0015
    min_agreement: float = 0.20
    profit_target: float = 0.012
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local-w2_1790981683'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.history_bars + params.hold_bars + params.max_streak + params.prior_split + 2)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        change = close.diff()
        direction = pd.Series(np.sign(change.fillna(0)), index=data.index).astype(int)
        groups = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(groups).cumcount().add(1).where(direction.ne(0), 0)
        run_end = direction.ne(direction.shift())
        previous = streak.shift().where(run_end).ffill().fillna(0)
        previous_direction = direction.shift().where(run_end).ffill().fillna(0)
        eligible = direction.ne(0) & previous_direction.eq(-direction) & streak.between(2, params.max_streak)
        prior_long = previous.gt(params.prior_split).astype(int)
        shape = ((direction.gt(0).astype(int) * (params.max_streak + 1) + streak.clip(upper=params.max_streak)) * 2 + prior_long).where(eligible)
        matured_shape = shape.shift(params.hold_bars)
        entry_open = data['open'].shift(params.hold_bars - 1)
        realized = (close / entry_open.where(entry_open > 0) - 1).replace([np.inf, -np.inf], np.nan)
        edge = pd.Series(np.nan, index=data.index, dtype=float)
        agreement = pd.Series(np.nan, index=data.index, dtype=float)
        count = pd.Series(0.0, index=data.index)
        for sign_bucket in (0, 1):
            for length in range(2, params.max_streak + 1):
                for prior_bucket in (0, 1):
                    key = (sign_bucket * (params.max_streak + 1) + length) * 2 + prior_bucket
                    member = matured_shape.eq(key) & realized.notna()
                    samples = member.astype(float).rolling(params.history_bars, min_periods=params.history_bars).sum()
                    total = realized.where(member, 0.0).rolling(params.history_bars, min_periods=params.history_bars).sum()
                    absolute = realized.abs().where(member, 0.0).rolling(params.history_bars, min_periods=params.history_bars).sum()
                    mean = total / samples.replace(0, np.nan)
                    consistency = (total.abs() / absolute.replace(0, np.nan)).clip(0, 1)
                    current = shape.eq(key)
                    edge = edge.where(~current, mean)
                    agreement = agreement.where(~current, consistency)
                    count = count.where(~current, samples)
        out['direction'] = direction
        out['streak'] = streak
        out['previous_streak'] = previous
        out['shape'] = shape
        out['matured_shape'] = matured_shape
        out['realized_return'] = realized
        out['analog_count'] = count.fillna(0)
        out['edge'] = edge.fillna(0)
        out['agreement'] = agreement.fillna(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        qualified = (indicators['analog_count'] >= params.min_analogs) & (indicators['edge'].abs() >= params.min_edge) & (indicators['agreement'] >= params.min_agreement)
        candidates = np.where(qualified, np.sign(indicators['edge']), 0).astype(int)
        conviction = (indicators['agreement'] / 0.60).clip(0, 1)
        sizes = (params.min_size + (params.max_size - params.min_size) * conviction).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        held_sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        position_size = params.min_size
        for i in range(len(data)):
            if position == 0 and i > 0 and signals[i - 1] != 0:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                position = int(signals[i - 1])
                entry_price = opens[i]
                entry_bar = i
                position_size = held_sizes[i - 1]
            if position != 0:
                gain = position * (closes[i] / entry_price - 1) if np.isfinite(closes[i]) else -np.inf
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    continue
                signals[i] = position
                held_sizes[i] = position_size
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = candidates[i]
                held_sizes[i] = sizes[i]
        df = pd.DataFrame({'signal': signals, 'size': held_sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
