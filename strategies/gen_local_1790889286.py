from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRecoveryParams:
    lookback: int = 7
    atr_window: int = 14
    min_drawdown: float = 0.015
    shock_share: float = 0.60
    recovery_fraction: float = 0.20
    volume_drought: float = 0.85
    volume_expansion: float = 1.20
    breakeven_pct: float = 0.004
    trail_atr: float = 1.25
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[ShockRecoveryParams]):
    strategy_id = 'gen_local_1790889286'

    @classmethod
    def params_type(cls) -> type[ShockRecoveryParams]:
        return ShockRecoveryParams

    def warmup_bars(self, params: ShockRecoveryParams) -> int:
        return int(max(params.lookback + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShockRecoveryParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_window < 1:
            raise ValueError('Lookbacks must be positive; lookback must be at least two.')
        if not (0 < params.shock_share <= 1 and params.min_drawdown > 0 and params.recovery_fraction > 0):
            raise ValueError('Invalid return thresholds.')
        if not (0 < params.volume_drought < 1 < params.volume_expansion):
            raise ValueError('Invalid volume thresholds.')
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.max_hold not in (1, 2):
            raise ValueError('Invalid exit parameters.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        prior = out['return'].shift(1)
        losses = -prior.clip(upper=0)
        out['return_debt'] = -prior.rolling(params.lookback, min_periods=params.lookback).sum()
        out['loss_sum'] = losses.rolling(params.lookback, min_periods=params.lookback).sum()
        out['largest_loss'] = losses.rolling(params.lookback, min_periods=params.lookback).max()
        out['shock_share'] = out['largest_loss'] / out['loss_sum'].where(out['loss_sum'] > 0)
        out['debt_fraction'] = out['return'] / out['return_debt'].where(out['return_debt'] > 0)
        out['previous_return'] = out['return'].shift(1)
        volume = data['volume'].where(data['volume'] >= 0)
        out['prior_volume_mean'] = volume.shift(1).rolling(params.lookback, min_periods=params.lookback).mean()
        out['volume_ratio'] = volume / out['prior_volume_mean'].where(out['prior_volume_mean'] > 0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        shock = (out['return_debt'] >= params.min_drawdown) & (out['shock_share'] >= params.shock_share)
        recovering = (out['debt_fraction'] >= params.recovery_fraction) & (out['debt_fraction'] < 1)
        relapsing = (out['debt_fraction'] <= -params.recovery_fraction) & (out['previous_return'] > 0)
        long_entry = shock & recovering & (out['volume_ratio'] <= params.volume_drought)
        short_entry = shock & relapsing & (out['volume_ratio'] >= params.volume_expansion)
        ready = np.isfinite(out['atr']) & (out['atr'] > 0) & np.isfinite(close)
        out['entry_direction'] = np.select([ready & long_entry, ready & short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockRecoveryParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        side = 0
        age = 0
        entry = 0.0
        stop = 0.0
        armed = False
        for i in range(len(data)):
            if side:
                if age == 0:
                    # The previous close's decision enters at this session's open.
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]):
                        side = 0
                        continue
                    entry = opens[i]
                    stop = -params.trail_atr * atr[i - 1] / entry
                    armed = False
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]):
                    side = 0
                    age = 0
                    continue
                gain = side * (closes[i] / entry - 1.0)
                if gain <= stop or age >= params.max_hold:
                    side = 0
                    age = 0
                    # Exit decisions execute next session; do not reenter on this close.
                    continue
                if gain >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-return stop only rises; the corresponding short price stop falls.
                    stop = max(stop, 0.0, gain - params.trail_atr * atr[i] / entry)
            elif entries[i] in (-1, 1):
                side = int(entries[i])
                age = 0
            decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [102.0] * 7,
        'low': [98.0] * 7,
        'close': [100.0, 101.0, 100.5, 100.0, 99.0, 99.8, 100.0],
        'volume': [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({'atr': 0.1, 'entry_direction': [1, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, ShockRecoveryParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result['size'] > 0).all()
    expiry = strategy.generate_signals(data, indicators, None, ShockRecoveryParams(max_hold=1)).data
    assert expiry['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
