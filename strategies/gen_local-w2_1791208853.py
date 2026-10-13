from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBreakoutParams:
    lookback: int = 7
    depth_floor: float = 0.01

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError('lookback must be at least 2')
        if not 0.0 < self.depth_floor < 1.0:
            raise ValueError('depth_floor must be between zero and one')


class GeneratedStrategy(BaseStrategy[DrawdownBreakoutParams]):
    strategy_id = 'gen_local-w2_1791208853'

    @classmethod
    def params_type(cls) -> type[DrawdownBreakoutParams]:
        return DrawdownBreakoutParams

    @staticmethod
    def warmup_bars(params: DrawdownBreakoutParams) -> int:
        return 4 * params.lookback + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownBreakoutParams) -> pd.DataFrame:
        n = params.lookback
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        out = pd.DataFrame(index=data.index)
        out['peak'] = close.rolling(3 * n, min_periods=3 * n).max().shift(1)
        out['previous_peak'] = out['peak'].shift(1)
        out['depth'] = 1.0 - close / out['peak']
        previous_depth = out['depth'].shift(1)
        out['depth_lower'] = previous_depth.rolling(n, min_periods=n).min()
        out['depth_upper'] = previous_depth.rolling(n, min_periods=n).max()
        out['ready'] = out[['peak', 'previous_peak', 'depth', 'depth_lower', 'depth_upper']].notna().all(axis=1)
        out['stable_peak'] = out['peak'].eq(out['previous_peak'])
        out['long_entry'] = (
            out['ready']
            & out['stable_peak']
            & out['depth'].lt(out['depth_lower'])
            & out['depth_upper'].ge(params.depth_floor)
        ).fillna(False)
        out['short_entry'] = (
            out['ready']
            & out['stable_peak']
            & out['depth'].gt(out['depth_upper'])
            & out['depth'].ge(params.depth_floor)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownBreakoutParams,
    ) -> SignalFrame:
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1:
                if shorts[i]:
                    position = 0
            elif position == -1:
                if longs[i]:
                    position = 0
            elif longs[i]:
                position = 1
            elif shorts[i]:
                position = -1
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'long_entry': [True, False, False, False, True, True],
        'short_entry': [False, False, True, True, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, DrawdownBreakoutParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, 0]
    assert result.data['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
