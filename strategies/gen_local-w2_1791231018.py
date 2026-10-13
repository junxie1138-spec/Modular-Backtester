from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    fractional_order: float = 0.65
    fractional_window: int = 32
    recent_window: int = 4
    reference_window: int = 24
    atr_window: int = 14
    entry_threshold: float = 2.0
    reset_threshold: float = 0.7
    trail_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = 'gen_local-w2_1791231018'

    @classmethod
    def params_type(cls) -> type[ChangeParams]:
        return ChangeParams

    def warmup_bars(self, params: ChangeParams) -> int:
        return int(max(params.fractional_window + params.recent_window + params.reference_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if min(params.fractional_window, params.recent_window, params.reference_window, params.atr_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if not 0.0 <= params.reset_threshold < params.entry_threshold:
            raise ValueError('thresholds must satisfy 0 <= reset < entry')
        if params.trail_atr <= 0.0 or params.hold_bars < 1:
            raise ValueError('trail_atr and hold_bars must be positive')
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase fractional_window if residual drift dominates.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        recent_mean = fractional.rolling(params.recent_window).mean()
        reference = fractional.shift(params.recent_window)
        reference_mean = reference.rolling(params.reference_window).mean()
        reference_std = reference.rolling(params.reference_window).std(ddof=0).replace(0.0, np.nan)
        contrast_scale = np.sqrt(1.0 / params.recent_window + 1.0 / params.reference_window)
        change_score = (recent_mean - reference_mean) / (reference_std * contrast_scale)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            'fractional_close': fractional,
            'recent_mean': recent_mean,
            'reference_mean': reference_mean,
            'reference_std': reference_std,
            'change_score': change_score,
            'previous_close': previous_close,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ChangeParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        scores = indicators['change_score'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side, age, armed = 0, 0, True
        high, low = -np.inf, np.inf
        stop, short_stop = -np.inf, np.inf
        for i, (price, score, volatility) in enumerate(zip(close, scores, atr)):
            if not np.isfinite([price, score, volatility]).all() or price <= 0.0 or volatility <= 0.0:
                side, age, armed = 0, 0, True
                continue
            if abs(score) <= params.reset_threshold:
                armed = True
            if side:
                age += 1
                high, low = max(high, price), min(low, price)
                # Highest-close stop applies to both directions and never falls.
                stop = max(stop, high - params.trail_atr * volatility)
                # Shorts also receive a mirrored adverse-price trailing stop.
                short_stop = min(short_stop, low + params.trail_atr * volatility)
                if (price <= stop
                    or (side == -1 and price >= short_stop)
                    or age >= params.hold_bars
                    or side * score <= -params.reset_threshold):
                    side, age = 0, 0
                    continue
            elif armed and abs(score) >= params.entry_threshold:
                side = 1 if score > 0.0 else -1
                age, armed = 0, False
                # Begin tracking closes on the next bar, when entry executes.
                high, low = -np.inf, np.inf
                stop, short_stop = -np.inf, np.inf
            decisions[i] = side
            confidence = (abs(score) - params.reset_threshold) / (2.0 * params.entry_threshold - params.reset_threshold)
            sizes[i] = float(np.clip(0.25 + 0.75 * confidence, 0.25, 1.0))
        df = pd.DataFrame({'signal': decisions, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ChangeParams()
    index = pd.date_range('2020-01-01', periods=5)
    for closes, score, expected in [
        ([100.0, 100.0, 105.0, 101.0, 101.0], 3.0, [0, 1, 1, 1, 0]),
        ([100.0, 100.0, 99.0, 103.0, 103.0], -3.0, [0, -1, -1, -1, 0]),
    ]:
        data = pd.DataFrame({'close': closes}, index=index)
        indicators = pd.DataFrame({'change_score': score, 'atr': 1.0}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == expected
        assert np.isfinite(result['size']).all() and (result['size'] > 0.0).all()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'close': 100.0}, index=index)
    indicators = pd.DataFrame({'change_score': 3.0, 'atr': 1.0}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 3


if __name__ == '__main__':
    _self_check()
