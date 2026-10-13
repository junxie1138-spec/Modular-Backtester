from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 20
    hub_multiple: float = 1.8
    position_min: float = 0.20
    position_max: float = 0.55
    hold_bars: int = 8

    def __post_init__(self):
        if not isinstance(self.window, int) or self.window < 5:
            raise ValueError('window must be an integer >= 5')
        if not isinstance(self.hold_bars, int) or self.hold_bars < 1:
            raise ValueError('hold_bars must be a positive integer')
        if not np.isfinite(self.hub_multiple) or self.hub_multiple <= 1:
            raise ValueError('hub_multiple must be finite and > 1')
        if not 0 <= self.position_min < self.position_max <= 1:
            raise ValueError('position bounds must satisfy 0 <= min < max <= 1')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791250074'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float).where(lambda s: np.isfinite(s) & (s > 0))
        width = params.window
        heights = -np.column_stack([
            close.shift(lag).to_numpy() for lag in range(width - 1, -1, -1)
        ])
        valid = np.isfinite(heights).all(axis=1)
        degrees = np.zeros(heights.shape, dtype=float)
        # ponytail: O(bars * window squared); use an incremental graph for large windows.
        for left in range(width - 1):
            blocker = np.full(len(data), -np.inf)
            for right in range(left + 1, width):
                edge = (
                    valid
                    & (blocker < np.minimum(heights[:, left], heights[:, right]))
                )
                degrees[:, left] += edge
                degrees[:, right] += edge
                blocker = np.maximum(blocker, heights[:, right])

        hub_index = np.argmax(np.where(np.isfinite(heights), heights, -np.inf), axis=1)
        rows = np.arange(len(data))
        out = pd.DataFrame(index=data.index)
        out['hub_degree'] = np.where(valid, degrees[rows, hub_index], np.nan)
        out['mean_degree'] = np.where(valid, degrees.mean(axis=1), np.nan)
        out['hub_multiple'] = out['hub_degree'] / out['mean_degree'].replace(0, np.nan)
        out['hub_age'] = np.where(valid, width - 1 - hub_index, np.nan)
        out['range_low'] = close.rolling(width, min_periods=width).min()
        out['range_high'] = close.rolling(width, min_periods=width).max()
        span = (out['range_high'] - out['range_low']).replace(0, np.nan)
        out['position'] = (close - out['range_low']) / span
        out['close_change'] = close.diff()
        out['eligible'] = (
            (out['hub_multiple'] >= params.hub_multiple)
            & out['hub_age'].between(2, width // 2)
            & out['position'].between(params.position_min, params.position_max)
            & (out['close_change'] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        eligible = indicators['eligible'].fillna(False).astype(bool)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for bar, enter in enumerate(confirmed.to_numpy()):
            if exit_at >= 0:
                if bar < exit_at:
                    signal[bar] = 1
                else:
                    # Reserve the exit bar for a flat target, even if entry remains confirmed.
                    exit_at = -1
            elif enter:
                signal[bar] = 1
                exit_at = bar + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = Params(hold_bars=7)
    index = pd.date_range('2020-01-01', periods=18)
    data = pd.DataFrame(
        {column: np.full(18, 100.0) for column in ('open', 'high', 'low', 'close', 'volume')},
        index=index,
    )
    indicators = pd.DataFrame({'eligible': True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 0] + [1] * 7 + [0] + [1] * 7 + [0]
    assert (result.data['size'] > 0).all()
    flat = GeneratedStrategy.indicators(None, data, params)
    assert not flat['eligible'].any()


if __name__ == '__main__':
    _self_check()
