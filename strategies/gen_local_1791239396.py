from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ComplexityParams:
    anchor_window: int = 63
    pattern_window: int = 20
    trough_window: int = 14
    atr_window: int = 14
    complexity_lag: int = 3
    complexity_floor: float = 0.08
    direction_bias: float = 0.15
    min_drawdown: float = 0.02
    max_drawdown: float = 0.15
    rebound_fraction: float = 0.01
    breakeven_gain: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[ComplexityParams]):
    strategy_id = 'gen_local_1791239396'

    @classmethod
    def params_type(cls) -> type[ComplexityParams]:
        return ComplexityParams

    def warmup_bars(self, params: ComplexityParams) -> int:
        return int(max(params.anchor_window, params.pattern_window + 2 + params.complexity_lag, params.trough_window + 1, params.atr_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: ComplexityParams) -> pd.DataFrame:
        p = params
        if min(p.anchor_window, p.pattern_window, p.trough_window, p.atr_window) < 2 or p.complexity_lag < 1 or p.max_hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0 < p.min_drawdown < p.max_drawdown < 1 and 0 < p.direction_bias <= 1 and 0 < p.complexity_floor < 1 and p.rebound_fraction > 0 and p.breakeven_gain > 0 and p.trail_atr > 0):
            raise ValueError('Invalid thresholds.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        x, y, z = close.shift(2), close.shift(1), close
        valid = x.notna() & y.notna() & z.notna()
        # Equal prices retain chronological ordering.
        patterns = [
            (x <= y) & (y <= z),
            (x <= z) & (z < y),
            (y < x) & (x <= z),
            (y <= z) & (z < x),
            (z < x) & (x <= y),
            (z < y) & (y < x),
        ]
        out['pattern'] = pd.Series(np.select(patterns, range(6), default=np.nan), index=data.index).where(valid)
        probability_columns = []
        for code in range(6):
            name = f'pattern_probability_{code}'
            out[name] = out['pattern'].eq(code).astype(float).where(valid).rolling(p.pattern_window, min_periods=p.pattern_window).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        ready = probabilities.notna().all(axis=1)
        entropy_terms = -(probabilities * np.log(probabilities.where(probabilities > 0))).fillna(0.0)
        out['entropy'] = (entropy_terms.sum(axis=1) / np.log(6.0)).where(ready)
        out['disequilibrium'] = ((probabilities - 1.0 / 6.0).pow(2).sum(axis=1) * 6.0 / 5.0).where(ready)
        out['complexity'] = out['entropy'] * out['disequilibrium']
        out['complexity_change'] = out['complexity'].diff(p.complexity_lag)
        out['direction_bias'] = out['pattern_probability_0'] - out['pattern_probability_5']
        out['anchor_high'] = close.rolling(p.anchor_window, min_periods=p.anchor_window).max()
        out['trough'] = close.rolling(p.trough_window, min_periods=p.trough_window).min()
        out['drawdown'] = 1.0 - close / out['anchor_high'].where(out['anchor_high'] > 0)
        out['repair'] = close / out['trough'].where(out['trough'] > 0) - 1.0
        out['previous_repair'] = out['repair'].shift(1)
        out['two_bar_move'] = close.diff(2)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - close.shift(1)).abs(), (data['low'] - close.shift(1)).abs()], axis=1).max(axis=1).where(close.shift(1).notna())
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        eligible = out['drawdown'].between(p.min_drawdown, p.max_drawdown) & out['complexity'].ge(p.complexity_floor) & out['complexity_change'].gt(0) & out['atr'].gt(0)
        recovering = out['repair'].ge(p.rebound_fraction) & out['two_bar_move'].gt(0) & out['direction_bias'].ge(p.direction_bias)
        failing = out['previous_repair'].ge(p.rebound_fraction) & out['two_bar_move'].lt(0) & out['direction_bias'].le(-p.direction_bias)
        out['candidate'] = np.select([eligible & recovering, eligible & failing], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ComplexityParams) -> SignalFrame:
        df = pd.DataFrame({'signal': np.zeros(len(data), dtype=int), 'size': np.ones(len(data), dtype=float)}, index=data.index)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        previous_candidate = 0
        entry = np.nan
        signed_stop = np.nan
        armed = False
        entry_bar = -1
        start = self.warmup_bars(params) - 1
        for i in range(len(data)):
            if i < start:
                continue
            if direction != 0:
                if np.isnan(entry):
                    # Yesterday's entry decision fills at today's open.
                    if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        direction = 0
                        previous_candidate = 0
                        continue
                    entry = opens[i]
                    entry_bar = i
                    signed_stop = direction * entry - params.trail_atr * atr[i - 1]
                    armed = False
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    direction = 0
                    entry = np.nan
                    previous_candidate = 0
                    continue
                signed_close = direction * closes[i]
                # Stops are close-triggered; every exit fills next bar.
                stopped = signed_close <= signed_stop
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if stopped or expired:
                    direction = 0
                    entry = np.nan
                    previous_candidate = 0
                    continue
                if direction * (closes[i] - entry) / entry >= params.breakeven_gain:
                    armed = True
                if armed:
                    # Signed coordinates make the stop ratchet upward for both directions.
                    signed_stop = max(signed_stop, direction * entry, signed_close - params.trail_atr * atr[i])
                decisions[i] = direction
                continue
            candidate = int(candidates[i])
            if candidate != 0 and candidate == previous_candidate:
                direction = candidate
                entry = np.nan
                decisions[i] = direction
                previous_candidate = 0
            else:
                previous_candidate = candidate
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ComplexityParams(anchor_window=3, pattern_window=3, trough_window=3, atr_window=2, complexity_lag=1)
    for direction in (1, -1):
        close = np.array([100.0] * 9 + [100.0 + direction * 2.0, 100.0 + direction * 4.0, 100.0 + direction, 100.0, 100.0, 100.0])
        index = pd.date_range('2020-01-01', periods=len(close))
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        computed = strategy.indicators(data, params)
        assert computed.index.equals(data.index)
        indicators = pd.DataFrame({'atr': 1.0, 'candidate': direction}, index=index)
        signals = strategy.generate_signals(data, indicators, None, params).data
        assert signals['signal'].iloc[7] == 0
        assert signals['signal'].iloc[8] == direction
        assert signals['signal'].iloc[11] == direction
        assert signals['signal'].iloc[12] == 0
        assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
