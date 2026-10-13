from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_down_streak: int = 3
    recovery_streak: int = 2
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_repair: float = 0.20
    max_repair: float = 0.65


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791241092'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.atr_period + 1, params.min_down_streak + params.recovery_streak + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close']
        change = close.diff()
        down = change.lt(0)
        up = change.gt(0)
        down_count = down.astype(int).groupby((~down).cumsum()).cumsum()
        up_count = up.astype(int).groupby((~up).cumsum()).cumsum()
        first_down = down & ~down.shift(1, fill_value=False)
        first_up = up & ~up.shift(1, fill_value=False)
        decline_origin = close.shift(1).where(first_down).ffill()
        completed_down = down_count.shift(1).where(first_up).ffill()
        recovery_origin = decline_origin.where(first_up).ffill()
        recovery_bottom = close.shift(1).where(first_up).ffill()
        decline_depth = recovery_origin - recovery_bottom
        repair = ((close - recovery_bottom) / decline_depth.where(decline_depth.gt(0))).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return pd.DataFrame({
            'down_streak': down_count,
            'up_streak': up_count,
            'completed_down': completed_down,
            'decline_origin': decline_origin,
            'recovery_origin': recovery_origin,
            'recovery_bottom': recovery_bottom,
            'decline_depth': decline_depth,
            'repair': repair,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        eligible = (
            indicators['up_streak'].eq(params.recovery_streak)
            & indicators['completed_down'].ge(params.min_down_streak)
            & indicators['repair'].between(params.min_repair, params.max_repair)
            & indicators['atr'].gt(0)
        ).fillna(False)
        conviction = (indicators['up_streak'] / indicators['completed_down'].where(indicators['completed_down'].gt(0))).clip(0, 1)
        entry_size = (0.95 * conviction).clip(0.20, 0.95).fillna(0.20)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        triggers = eligible.to_numpy(dtype=bool)
        weights = entry_size.to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.20, dtype=float)
        active = False
        pending = False
        entry_bar = 0
        stop = 0.0
        position_size = 0.20
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                size[i] = position_size
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
            elif triggers[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position_size = float(weights[i])
                size[i] = position_size
                signal[i] = 1
                pending = True
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0, 110.0, 109.0, 108.0, 105.0, 105.0],
        'high': [101.0, 111.0, 110.0, 109.0, 106.0, 106.0],
        'low': [99.0, 108.0, 107.0, 104.0, 104.0, 104.0],
        'close': [100.0, 109.0, 108.0, 105.0, 105.0, 105.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'up_streak': [2, 3, 4, 0, 0, 0],
        'completed_down': [4] * 6,
        'repair': [0.4] * 6,
        'atr': [2.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.isclose(result['size'].iloc[1], 0.475)
    assert result['size'].gt(0).all()
    expiry = GeneratedStrategy.generate_signals(data, indicators, None, RecoveryParams(stop_atr=100.0, hold_bars=2)).data
    assert expiry['signal'].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
