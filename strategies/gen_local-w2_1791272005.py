from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    trend_window: int = 6
    rank_window: int = 60
    atr_window: int = 14
    entry_percentile: float = 0.80
    stop_atr: float = 1.5
    hold_days: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = 'gen_local-w2_1791272005'

    @classmethod
    def params_type(cls) -> type[MigrationParams]:
        return MigrationParams

    @staticmethod
    def warmup_bars(params: MigrationParams) -> int:
        return int(max(params.trend_window + params.rank_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        if min(params.trend_window, params.rank_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError('Entry percentile must be between 0.5 and 1.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Stop ATR must be finite and positive.')
        if not 3 <= params.hold_days <= 5:
            raise ValueError('Holding period must be three to five days.')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')

        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        upward = data['high'].diff()
        downward = -data['low'].diff()
        valid_movement = upward.notna() & downward.notna()
        positive_movement = upward.where((upward > downward) & (upward > 0), 0.0).where(valid_movement)
        negative_movement = downward.where((downward > upward) & (downward > 0), 0.0).where(valid_movement)
        positive_sum = positive_movement.rolling(params.trend_window, min_periods=params.trend_window).sum()
        negative_sum = negative_movement.rolling(params.trend_window, min_periods=params.trend_window).sum()
        range_sum = true_range.rolling(params.trend_window, min_periods=params.trend_window).sum().replace(0.0, np.nan)
        dominance = ((positive_sum - negative_sum) / range_sum).clip(-1.0, 1.0)
        strength = dominance.abs()
        percentile = strength.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            'previous_close': previous_close,
            'true_range': true_range,
            'upward': upward,
            'downward': downward,
            'positive_movement': positive_movement,
            'negative_movement': negative_movement,
            'positive_sum': positive_sum,
            'negative_sum': negative_sum,
            'range_sum': range_sum,
            'dominance': dominance,
            'strength': strength,
            'percentile': percentile,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        dominance = indicators['dominance'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        percentile = indicators['percentile'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        valid = np.isfinite(dominance) & np.isfinite(strength) & np.isfinite(percentile) & np.isfinite(atr) & (atr > 0)
        candidates = np.where(valid & (percentile >= params.entry_percentile), np.sign(dominance), 0).astype(int)
        rank_conviction = np.clip((percentile - params.entry_percentile) / (1.0 - params.entry_percentile), 0.0, 1.0)
        conviction = np.nan_to_num(rank_conviction * np.clip(strength, 0.0, 1.0), nan=0.0)
        proposed_size = params.min_size + (params.max_size - params.min_size) * conviction
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        age = 0
        entry_atr = 0.0
        fixed_stop = np.nan
        trade_size = params.min_size

        for i in range(len(data)):
            if position != 0:
                # The previous close's entry decision fills at this bar's open.
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                    else:
                        fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                if position != 0:
                    stopped = not np.isfinite(closes[i]) or position * (closes[i] - fixed_stop) < 0
                    if stopped or age >= params.hold_days:
                        position = 0
                # Exit decisions remain flat for the next execution bar.
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                age = 0
                entry_atr = float(atr[i])
                trade_size = float(proposed_size[i])
            signals[i] = position
            sizes[i] = trade_size if position != 0 else params.min_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = MigrationParams()
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({
        'open': 100.0,
        'high': 102.0,
        'low': 98.0,
        'close': [100.0, 100.0, 99.0, 100.0, 100.0, 100.0, 101.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'dominance': [0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
        'strength': 0.5,
        'percentile': [0.85, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
        'atr': 0.5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    assert result['size'].iloc[5] > result['size'].iloc[1]
    indicators['dominance'] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data['close'] = 100.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert calculated['percentile'].isna().all()


if __name__ == '__main__':
    _self_check()
