from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.5
    fractional_window: int = 32
    position_window: int = 20
    reclaim: float = 0.2
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791311450'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fractional_window + params.position_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.fractional_order < 1 and params.fractional_window >= 2 and params.position_window >= 2 and params.atr_window >= 1 and 0 < params.reclaim < 0.5 and params.breakeven_pct > 0 and params.trail_atr > 0 and 3 <= params.hold_days <= 5):
            raise ValueError('Invalid strategy parameters')
        close = data['close'].astype(float)
        fd = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fd = fd + weight * close.shift(lag)
        lower = fd.rolling(params.position_window).min()
        upper = fd.rolling(params.position_window).max()
        previous = close.shift(1)
        tr = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        tr = tr.where(previous.notna())
        return pd.DataFrame({
            'fractional_close': fd,
            'fractional_change': fd.diff(),
            'relative_position': (fd - lower) / (upper - lower).replace(0, np.nan),
            'ma200': close.rolling(200).mean(),
            'atr': tr.rolling(params.atr_window).mean(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        position = indicators['relative_position']
        previous = position.shift(1)
        long_entry = (previous < params.reclaim) & (position >= params.reclaim) & (indicators['fractional_change'] > 0) & (data['close'] > indicators['ma200'])
        short_entry = (previous > 1 - params.reclaim) & (position <= 1 - params.reclaim) & (indicators['fractional_change'] < 0) & (data['close'] < indicators['ma200'])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        valid = np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1) & np.isfinite(close) & (close > 0) & (atr > 0)
        signal = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        pending, armed = False, False
        entry, stop, peak = 0.0, 0.0, 0.0
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0 or i == 0 or not np.isfinite(atr[i - 1]):
                    side = 0
                    continue
                # Resolve the prior close's entry at today's open.
                entry = opens[i]
                peak = side * entry
                stop = peak - params.trail_atr * atr[i - 1]
                age, armed = 0, False
            if side:
                age += 1
                if not valid[i]:
                    side = 0
                    continue
                # Signed prices let both long and short stop protection ratchet upward.
                # Stops are checked at closes; exit orders execute the following bar.
                signed_close = side * close[i]
                regime_lost = side * (close[i] - ma[i]) <= 0
                if signed_close <= stop or regime_lost or age >= params.hold_days:
                    side = 0
                    continue
                peak = max(peak, signed_close)
                armed = armed or (signed_close - side * entry) / entry >= params.breakeven_pct
                if armed:
                    stop = max(stop, side * entry, peak - params.trail_atr * atr[i])
                signal[i] = side
            elif valid[i] and candidates[i]:
                side = int(candidates[i])
                pending = True
                signal[i] = side
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy, params = GeneratedStrategy(), Params()
    idx = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': [100.0, 100.0, 101.1, 100.5, 99.9, 100.0, 100.0], 'volume': 1000.0}, index=idx)
    ind = pd.DataFrame({'fractional_close': 1.0, 'fractional_change': 1.0, 'relative_position': [0.1, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4], 'ma200': 90.0, 'atr': 1.0}, index=idx)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0]
    data['close'] = 200 - data['close']
    ind['ma200'] = 110.0
    ind['relative_position'] = 1 - ind['relative_position']
    ind['fractional_change'] = -1.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result['signal'].tolist() == [0, 0, -1, -1, -1, 0, 0]
