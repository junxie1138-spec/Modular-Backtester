from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeGapParams:
    compression_bars: int = 5
    baseline_bars: int = 20
    compression_ratio: float = 0.70
    min_gap: float = 0.002
    min_body_efficiency: float = 0.45
    profit_target: float = 0.012
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeGapParams]):
    strategy_id = 'gen_local-w2_1790973659'

    @classmethod
    def params_type(cls) -> type[RangeGapParams]:
        return RangeGapParams

    def warmup_bars(self, params: RangeGapParams) -> int:
        return int(max(params.compression_bars, params.baseline_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeGapParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data['close'].replace(0.0, np.nan)
        daily_range = (data['high'] - data['low']).clip(lower=0.0)
        result['relative_range'] = daily_range / close
        result['recent_range'] = result['relative_range'].rolling(params.compression_bars).mean().shift(1)
        result['baseline_range'] = result['relative_range'].rolling(params.baseline_bars).mean().shift(1)
        result['compression'] = result['recent_range'] / result['baseline_range'].replace(0.0, np.nan)
        result['box_high'] = data['high'].rolling(params.compression_bars).max().shift(1)
        result['box_low'] = data['low'].rolling(params.compression_bars).min().shift(1)
        result['gap'] = data['open'] / close.shift(1) - 1.0
        result['body_efficiency'] = (data['close'] - data['open']) / daily_range.replace(0.0, np.nan)
        result['gap_vote'] = np.select(
            [(result['gap'] >= params.min_gap) & (data['open'] > result['box_high']),
             (result['gap'] <= -params.min_gap) & (data['open'] < result['box_low'])],
            [1, -1], default=0
        ).astype(int)
        result['body_vote'] = np.select(
            [result['body_efficiency'] >= params.min_body_efficiency,
             result['body_efficiency'] <= -params.min_body_efficiency],
            [1, -1], default=0
        ).astype(int)
        result['entry'] = np.where(
            (result['compression'] <= params.compression_ratio)
            & (result['gap_vote'] != 0)
            & (result['gap_vote'] == result['body_vote']),
            result['gap_vote'], 0
        ).astype(int)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeGapParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if position != 0:
                # The previous close's instruction fills at this bar's open.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = (
                    valid_entry and np.isfinite(closes[i])
                    and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                )
                if not valid_entry or target_hit or held_bars >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
            elif entries[i] != 0:
                position = int(entries[i])
                held_bars = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = RangeGapParams()
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    for side in (1, -1):
        for hit_target in (False, True):
            data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0,
                                 'close': 100.0, 'volume': 1000.0}, index=index)
            if hit_target:
                data.iloc[2, data.columns.get_loc('close')] = 100.0 + side * 2.0
            indicators = pd.DataFrame({'entry': [side, 0, 0, 0, 0, 0, 0]}, index=index)
            actual = strategy.generate_signals(data, indicators, None, params).data['signal'].tolist()
            expected = [0, side, side, 0, 0, 0, 0] if hit_target else [0, side, side, side, side, 0, 0]
            assert actual == expected, (actual, expected)
