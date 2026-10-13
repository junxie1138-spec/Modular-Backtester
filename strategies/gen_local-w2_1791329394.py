from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AcceptanceParams:
    percentile_window: int = 60
    shock_quantile: float = 0.85
    contraction_quantile: float = 0.40
    atr_window: int = 14
    breakeven_pct: float = 0.005
    trail_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AcceptanceParams]):
    strategy_id = 'gen_local-w2_1791329394'

    @classmethod
    def params_type(cls) -> type[AcceptanceParams]:
        return AcceptanceParams

    def warmup_bars(self, params: AcceptanceParams) -> int:
        return int(max(params.percentile_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: AcceptanceParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.atr_window < 1:
            raise ValueError('Lookback windows must be positive; percentile_window must exceed one.')
        if not (0.0 < params.shock_quantile < 1.0 and 0.0 < params.contraction_quantile < 1.0):
            raise ValueError('Quantiles must lie strictly between zero and one.')
        if params.trail_atr <= 0.0 or params.breakeven_pct <= 0.0 or params.hold_bars not in (1, 2):
            raise ValueError('Stops must be positive and hold_bars must be one or two.')

        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(lambda x: x > 0.0)
        out['range_threshold'] = out['range'].rolling(params.percentile_window).quantile(params.shock_quantile).shift(1)
        out['contraction'] = out['range'] / out['range'].shift(1)
        out['contraction_threshold'] = out['contraction'].rolling(params.percentile_window).quantile(params.contraction_quantile).shift(1)
        out['shock'] = (out['range'] > out['range_threshold']).astype(int)
        out['previous_midpoint'] = (data['high'].shift(1) + data['low'].shift(1)) / 2.0
        out['previous_body'] = data['close'].shift(1) - data['open'].shift(1)

        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()

        confirmation = (
            out['shock'].shift(1).eq(1)
            & out['contraction'].le(out['contraction_threshold'])
            & out['atr'].gt(0.0)
        )
        long_entry = confirmation & out['previous_body'].gt(0.0) & data['low'].gt(out['previous_midpoint'])
        short_entry = confirmation & out['previous_body'].lt(0.0) & data['high'].lt(out['previous_midpoint'])
        out['candidate'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AcceptanceParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        direction = 0
        pending = False
        entry_bar = -1
        entry_price = 0.0
        stop = 0.0
        best = 0.0
        armed = False

        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and opening > 0.0 and low > 0.0 and low <= close <= high and low <= opening <= high
            if not valid:
                direction = 0
                pending = False
                continue

            if pending:
                # Yesterday's decision fills at today's open.
                entry_price = opening
                entry_bar = i
                best = direction * entry_price
                stop = best - params.trail_atr * atr[i - 1]
                armed = False
                pending = False

            if direction != 0:
                # Signed prices let the protective stop ratchet upward for both directions.
                adverse = low if direction == 1 else -high
                favorable = high if direction == 1 else -low
                expired = i - entry_bar + 1 >= params.hold_bars
                if adverse <= stop or expired or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    direction = 0
                    continue

                best = max(best, favorable)
                if best - direction * entry_price >= entry_price * params.breakeven_pct:
                    armed = True
                if armed:
                    # A stop established at this close is checked from the next session onward.
                    stop = max(stop, direction * entry_price, best - params.trail_atr * atr[i])
                raw[i] = direction
                continue

            if candidates[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0.0:
                direction = int(candidates[i])
                pending = True
                raw[i] = direction

        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=4, freq='D')
    data = pd.DataFrame({
        'open': [100.0, 100.0, 100.5, 100.0],
        'high': [100.5, 101.0, 100.8, 100.5],
        'low': [99.5, 99.5, 99.9, 99.5],
        'close': [100.0, 100.8, 100.2, 100.0],
        'volume': [1000.0] * 4,
    }, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 4, 'candidate': [1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = AcceptanceParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    indicators['candidate'] = [-1, 0, 0, 0]
    mirrored = data.copy()
    mirrored['open'] = 200.0 - data['open']
    mirrored['high'] = 200.0 - data['low']
    mirrored['low'] = 200.0 - data['high']
    mirrored['close'] = 200.0 - data['close']
    result = strategy.generate_signals(mirrored, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0]
    assert result['size'].gt(0.0).all()
    warmup = strategy.indicators(data, params)
    assert warmup['candidate'].eq(0).all()


if __name__ == '__main__':
    _self_check()
