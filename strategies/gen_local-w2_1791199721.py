from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    range_lookback: int = 10
    atr_lookback: int = 14
    compression_ratio: float = 0.75
    gap_retention: float = 0.70
    minimum_gap_atr: float = 0.15
    maximum_gap_atr: float = 1.20
    breakeven_pct: float = 0.004
    initial_stop_atr: float = 1.0
    trail_atr: float = 0.75
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = 'gen_local-w2_1791199721'

    @classmethod
    def params_type(cls):
        return GapAcceptanceParams

    def warmup_bars(self, params: GapAcceptanceParams) -> int:
        return int(max(params.range_lookback, params.atr_lookback) + 1)

    def indicators(self, data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['previous_high'] = data['high'].shift(1)
        out['range'] = data['high'] - data['low']
        out['reference_range'] = out['range'].shift(1).rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).median()
        components = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1)
        out['true_range'] = components.max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_lookback, min_periods=params.atr_lookback
        ).mean()
        out['previous_atr'] = out['atr'].shift(1)
        out['gap'] = data['open'] - out['previous_close']
        out['gap_atr'] = out['gap'].div(out['previous_atr'].where(out['previous_atr'] > 0))
        out['retention'] = (data['low'] - out['previous_close']).div(
            out['gap'].where(out['gap'] > 0)
        )
        out['compressed'] = (
            (out['range'] > 0)
            & (out['reference_range'] > 0)
            & (out['range'] <= params.compression_ratio * out['reference_range'])
        )
        out['gap_accepted'] = (
            (data['open'] > out['previous_high'])
            & (data['low'] > out['previous_high'])
            & (out['retention'] >= params.gap_retention)
            & out['gap_atr'].between(params.minimum_gap_atr, params.maximum_gap_atr)
        )
        out['entry'] = (out['compressed'] & out['gap_accepted']).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAcceptanceParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        pending = False
        holding = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        age = 0

        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if pending:
                pending = False
                if i > 0 and np.isfinite(opening) and opening > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    holding = True
                    armed = False
                    age = 0
                    entry_price = opening
                    stop = entry_price - params.initial_stop_atr * atr[i - 1]

            if holding:
                age += 1
                if not np.isfinite(prices[i]).all() or not np.isfinite(atr[i]) or atr[i] <= 0:
                    holding = False
                    continue
                if low <= stop:
                    holding = False
                    continue
                if high >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, high - params.trail_atr * atr[i])
                    # ponytail: daily OHLC cannot order extrema; assume high precedes low when a new stop is touched.
                    if low <= stop:
                        holding = False
                        continue
                if age >= params.max_hold_bars:
                    holding = False
                    continue
                raw[i] = 1
            elif entries[i] and np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
                pending = True

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        # All exit decisions, including stop touches, execute at the following open.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 100.6, 100.0],
        'high': [100.2, 101.0, 101.1, 100.2],
        'low': [99.9, 100.4, 100.1, 99.8],
        'close': [100.0, 100.8, 100.5, 100.0],
        'volume': [1000.0] * 4,
    }, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 4, 'entry': [True, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = GapAcceptanceParams(max_hold_bars=2)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    assert result['size'].gt(0).all()
    params.max_hold_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]


if __name__ == '__main__':
    _self_check()
