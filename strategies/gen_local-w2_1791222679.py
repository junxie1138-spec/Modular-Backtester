from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidenceParams:
    peak_window: int = 63
    block_window: int = 5
    depth_threshold: float = 0.03
    occupancy_extreme: float = 0.8
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ResidenceParams]):
    strategy_id = 'gen_local-w2_1791222679'

    @classmethod
    def params_type(cls) -> type[ResidenceParams]:
        return ResidenceParams

    def warmup_bars(self, params: ResidenceParams) -> int:
        return int(max(200, params.peak_window + 2 * params.block_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ResidenceParams) -> pd.DataFrame:
        if min(params.peak_window, params.block_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError('Window lengths and holding period must be positive.')
        if not (0 < params.depth_threshold < 1 and 0.5 < params.occupancy_extreme <= 1):
            raise ValueError('Invalid drawdown or occupancy threshold.')
        if not (0 < params.breakeven_pct < 1 and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError('Invalid exit parameters.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['depth'] = (1.0 - close / out['peak'].where(out['peak'] > 0)).clip(lower=0.0)
        out['deep'] = (out['depth'] >= params.depth_threshold).astype(float).where(out['depth'].notna())
        out['occupancy'] = out['deep'].rolling(params.block_window, min_periods=params.block_window).mean()
        out['prior_occupancy'] = out['occupancy'].shift(params.block_window)
        out['anchor_unchanged'] = out['peak'].eq(out['peak'].shift(2 * params.block_window))
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        extreme = params.occupancy_extreme
        out['evacuation'] = ((out['prior_occupancy'] >= extreme) & (out['occupancy'] <= 1.0 - extreme + 1e-12) & out['anchor_unchanged'])
        out['recruitment'] = ((out['prior_occupancy'] <= 1.0 - extreme + 1e-12) & (out['occupancy'] >= extreme) & out['anchor_unchanged'])
        out['long_event'] = out['evacuation'] & ~out['evacuation'].shift(1, fill_value=False) & (close > out['ma200'])
        out['short_event'] = out['recruitment'] & ~out['recruitment'].shift(1, fill_value=False) & (close < out['ma200'])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResidenceParams) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        long_event = indicators['long_event'].fillna(False).to_numpy(dtype=bool)
        short_event = indicators['short_event'].fillna(False).to_numpy(dtype=bool)
        position = pending = age = 0
        entry = best = stop = 0.0
        activated = False
        for i in range(n):
            opening, high, low, close = prices[i]
            valid = bool(np.isfinite(prices[i]).all() and (prices[i] > 0).all() and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(ma[i]))
            if not valid:
                position = pending = 0
                continue
            if pending:
                # Yesterday's decision fills at today's open; no future open is read.
                position = pending
                pending = 0
                entry = opening
                age = 0
                best = 0.0
                activated = False
                stop = -params.trail_atr * atr[i - 1]
            if position:
                age += 1
                # Stops are observed from completed bars and exit next bar.
                # Signed profit coordinates make the stop ratchet upward for both sides.
                adverse = position * ((low if position == 1 else high) - entry)
                regime_ok = close > ma[i] if position == 1 else close < ma[i]
                if adverse <= stop or age >= params.hold_bars or not regime_ok:
                    position = 0
                    continue
                profit = position * (close - entry)
                best = max(best, profit)
                activated = activated or best >= entry * params.breakeven_pct
                if activated:
                    stop = max(stop, 0.0, best - params.trail_atr * atr[i])
                    if profit <= stop:
                        position = 0
                        continue
                raw[i] = position
            else:
                pending = 1 if long_event[i] else (-1 if short_event[i] else 0)
                raw[i] = pending
        df = pd.DataFrame({'signal': raw, 'size': np.ones(n, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ResidenceParams()
    index = pd.date_range('2020-01-01', periods=7)
    for side in (1, -1):
        close = 100.0 + side * np.array([0.0, 0.0, 3.0, 4.0, -1.0, 0.0, 0.0])
        data = pd.DataFrame({'open': 100.0, 'high': close + 0.1, 'low': close - 0.1, 'close': close, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 1.0, 'ma200': 90.0 if side == 1 else 110.0, 'long_event': False, 'short_event': False}, index=index)
        ind.loc[index[0], 'long_event' if side == 1 else 'short_event'] = True
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, side, 0, 0]
        assert (result['size'] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:4], ind.iloc[:4], None, params).data
        assert prefix.equals(result.iloc[:4])
    flat = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=250))
    ind = strategy.indicators(flat, params)
    assert strategy.warmup_bars(params) >= 200
    assert not strategy.generate_signals(flat, ind, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
