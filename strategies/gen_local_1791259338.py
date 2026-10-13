from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    signature_window: int = 8
    breakout_window: int = 12
    area_threshold: float = 0.12
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791259338'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if min(params.signature_window, params.breakout_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if params.max_hold < 1 or not 0 < params.area_threshold < 0.5:
            raise ValueError('Invalid holding horizon or area threshold.')
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError('Breakeven percentage must be positive and finite.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('ATR multiplier must be positive and finite.')
        return int(max(params.signature_window, params.breakout_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        previous = close.shift(1)
        out['gap'] = np.log(opening / previous)
        out['body'] = np.log(close / opening)
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Second-level antisymmetric signature of the local gap/body path.
        for lag in range(params.signature_window - 1, -1, -1):
            dx = out['gap'].shift(lag)
            dy = out['body'].shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
        out['gap_sum'] = x
        out['body_sum'] = y
        out['signature_area'] = area
        out['gap_variation'] = out['gap'].abs().rolling(params.signature_window).sum()
        out['body_variation'] = out['body'].abs().rolling(params.signature_window).sum()
        denominator = (out['gap_variation'] * out['body_variation']).replace(0.0, np.nan)
        out['normalized_area'] = area / denominator
        # Mirroring both coordinates preserves area; net gap gives direction.
        out['signature_direction'] = np.where(
            out['normalized_area'].ge(params.area_threshold), np.sign(x), 0
        ).astype(int)
        out['channel_high'] = data['high'].shift(1).rolling(params.breakout_window).max()
        out['channel_low'] = data['low'].shift(1).rolling(params.breakout_window).min()
        out['breakout_direction'] = np.select(
            [close.gt(out['channel_high']), close.lt(out['channel_low'])],
            [1, -1], default=0
        ).astype(int)
        out['true_range'] = pd.concat(
            [data['high'] - data['low'], (data['high'] - previous).abs(),
             (data['low'] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        ready = np.isfinite(out[['normalized_area', 'channel_high', 'channel_low', 'atr']]).all(axis=1)
        agreed = out['signature_direction'].eq(out['breakout_direction'])
        out['candidate'] = out['signature_direction'].where(ready & agreed & out['atr'].gt(0), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        self.warmup_bars(params)
        count = len(data)
        raw = np.zeros(count, dtype=int)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        side = 0
        entry = 0.0
        entered = 0
        stop = 0.0
        armed = False
        for i in range(count):
            # Yesterday's target takes effect at today's open.
            if side == 0 and i > 0 and raw[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                side = int(raw[i - 1])
                entry = opens[i]
                entered = i
                armed = False
                # Signed-price stop only increases; short price stops decrease.
                stop = side * entry - params.trail_atr * atr[i - 1]
            if side != 0:
                price = closes[i]
                if not np.isfinite(price) or price <= 0:
                    side = 0
                    continue
                signed_price = side * price
                if signed_price <= stop or i - entered + 1 >= params.max_hold:
                    side = 0
                    continue
                if side * (price / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, signed_price - params.trail_atr * atr[i])
                    if signed_price <= stop:
                        side = 0
                        continue
                raw[i] = side
            else:
                raw[i] = candidates[i]
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        # Stops are close-observed decisions executed at the following open.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=6)
    for side in (1, -1):
        close = 100.0 + side * np.array([0.0, 2.0, 4.0, 1.0, 0.0, 0.0])
        data = pd.DataFrame({
            'open': 100.0, 'high': close + 1.0, 'low': close - 1.0,
            'close': close, 'volume': 1000.0
        }, index=index)
        derived = pd.DataFrame({'candidate': [side, 0, 0, 0, 0, 0], 'atr': 1.0}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, 0, 0]
        assert result['size'].gt(0).all()
        cold = strategy.indicators(data, params)
        assert cold.index.equals(data.index)
        assert cold['candidate'].eq(0).all()
    assert strategy.warmup_bars(params) == 15


if __name__ == '__main__':
    _self_check()
