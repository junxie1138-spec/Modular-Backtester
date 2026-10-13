from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockAbsorptionParams:
    shock_multiple: float = 1.5
    compression_ratio: float = 0.55
    profit_target: float = 0.015
    max_hold: int = 4

    def __post_init__(self):
        if not np.isfinite(self.shock_multiple) or self.shock_multiple <= 1.0:
            raise ValueError('shock_multiple must exceed 1')
        if not np.isfinite(self.compression_ratio) or not 0.0 < self.compression_ratio < 1.0:
            raise ValueError('compression_ratio must be between 0 and 1')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be positive')
        if isinstance(self.max_hold, bool) or not isinstance(self.max_hold, int) or not 3 <= self.max_hold <= 5:
            raise ValueError('max_hold must be an integer from 3 to 5')


class GeneratedStrategy(BaseStrategy[ShockAbsorptionParams]):
    strategy_id = 'gen_local-w2_1791292983'

    @classmethod
    def params_type(cls):
        return ShockAbsorptionParams

    def warmup_bars(self, params: ShockAbsorptionParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: ShockAbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).clip(lower=0.0)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['baseline_range'] = out['range'].rolling(20, min_periods=20).mean().shift(5)
        out['shock_range'] = out['range'].shift(4)
        out['close_location'] = (data['close'] - data['low']) / out['range'].replace(0.0, np.nan)
        out['shock_bearish'] = (data['close'].shift(4) < data['open'].shift(4)) & (out['close_location'].shift(4) <= 0.35)
        out['compression_range'] = out['range'].rolling(3, min_periods=3).mean().shift(1)
        out['overlap_low'] = data['low'].rolling(3, min_periods=3).max().shift(1)
        out['overlap_high'] = data['high'].rolling(3, min_periods=3).min().shift(1)
        out['ceiling'] = data['high'].rolling(3, min_periods=3).max().shift(1)
        out['entry'] = (
            (data['close'] > out['ma200'])
            & (out['baseline_range'] > 0.0)
            & (out['shock_range'] >= params.shock_multiple * out['baseline_range'])
            & out['shock_bearish']
            & (out['compression_range'] <= params.compression_ratio * out['shock_range'])
            & (out['overlap_low'] <= out['overlap_high'])
            & (data['close'] > out['ceiling'])
            & (data['close'] > data['open'])
            & (out['range'] < out['shock_range'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockAbsorptionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        held = 0

        for i in range(len(data)):
            if active:
                # Yesterday's entry decision fills at today's open.
                if held == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                held += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.max_hold:
                    # This flat decision executes at the next session's open.
                    active = False
                    held = 0
                    entry_price = np.nan
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                held = 0
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = ShockAbsorptionParams()
    index = pd.date_range('2020-01-01', periods=8)
    sample = pd.DataFrame({column: 100.0 for column in ('open', 'high', 'low', 'close', 'volume')}, index=index)
    flags = pd.DataFrame({'entry': [True] + [False] * 7}, index=index)
    timed = strategy.generate_signals(sample, flags, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    sample.loc[index[2], 'close'] = 102.0
    targeted = strategy.generate_signals(sample, flags, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (targeted['size'] > 0.0).all()
