from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTakeoverParams:
    phase_bars: int = 5
    breakout_bars: int = 15
    takeover_share: float = 0.65
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.phase_bars, self.breakout_bars, self.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.5 < self.takeover_share < 1.0:
            raise ValueError('takeover_share must be between 0.5 and 1.0.')


class GeneratedStrategy(BaseStrategy[RangeTakeoverParams]):
    strategy_id = 'gen_local-w2_1791327877'

    @classmethod
    def params_type(cls) -> type[RangeTakeoverParams]:
        return RangeTakeoverParams

    def warmup_bars(self, params: RangeTakeoverParams) -> int:
        return int(max(2 * params.phase_bars + 1, params.breakout_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RangeTakeoverParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prices = data[['high', 'low', 'close']].replace([np.inf, -np.inf], np.nan)
        out['up_extension'] = prices['high'].diff().clip(lower=0.0)
        out['down_extension'] = (-prices['low'].diff()).clip(lower=0.0)
        out['up_total'] = out['up_extension'].rolling(params.phase_bars, min_periods=params.phase_bars).sum()
        out['down_total'] = out['down_extension'].rolling(params.phase_bars, min_periods=params.phase_bars).sum()
        out['extension_total'] = out['up_total'] + out['down_total']
        out['up_share'] = out['up_total'] / out['extension_total'].replace(0.0, np.nan)
        out['previous_share'] = out['up_share'].shift(params.phase_bars)
        out['channel_high'] = prices['high'].rolling(params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)
        out['takeover_strength'] = (out['up_share'] - out['previous_share']).clip(lower=0.25, upper=0.95).fillna(0.25)
        ready = out[['up_share', 'previous_share', 'channel_high']].notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['previous_share'] <= 1.0 - params.takeover_share)
            & (out['up_share'] >= params.takeover_share)
            & (prices['close'] > out['channel_high'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeTakeoverParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['takeover_strength'].replace([np.inf, -np.inf], np.nan).fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        exit_at = -1
        entry_size = 0.25
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif entries[i]:
                exit_at = i + params.hold_bars
                entry_size = float(strengths[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'entry': [False, True, True, True, True, True, False],
        'takeover_strength': [0.25, 0.7, 0.9, 0.9, 0.9, 0.5, 0.25],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RangeTakeoverParams(hold_bars=3))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 0, 1]
    assert result.data['size'].iloc[2:5].tolist() == [0.7, 0.7, 0.7]
    empty = GeneratedStrategy.generate_signals(None, data.iloc[:0], indicators.iloc[:0], None, RangeTakeoverParams())
    assert empty.data.empty


if __name__ == '__main__':
    _self_check()
