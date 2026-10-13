from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    percentile_window: int = 126
    depth_percentile: float = 0.25
    streak_percentile: float = 0.90
    hold_bars: int = 18

    def __post_init__(self):
        if self.peak_window < 2 or self.percentile_window < 2 or self.hold_bars < 1:
            raise ValueError('Lookbacks must be at least two and hold_bars positive.')
        if not 0.0 < self.depth_percentile < 1.0 or not 0.0 < self.streak_percentile < 1.0:
            raise ValueError('Percentiles must lie strictly between zero and one.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791243522'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(params.peak_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        low = data['low'].where(data['low'] > 0)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = close.div(out['peak']).sub(1.0)
        out['depth_threshold'] = out['drawdown'].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.depth_percentile).shift(1)
        out['close_change'] = close.diff()
        out['low_change'] = low.diff()
        out['recovering'] = (out['close_change'] > 0) & (out['low_change'] > 0)
        out['streak_group'] = (~out['recovering']).cumsum()
        out['streak'] = out['recovering'].astype(int).groupby(out['streak_group']).cumsum()
        out['streak_threshold'] = out['streak'].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.streak_percentile).shift(1)
        # Record depth and its historical threshold immediately before each streak.
        out['origin_drawdown'] = out['drawdown'].where(~out['recovering']).ffill()
        out['origin_threshold'] = out['depth_threshold'].where(~out['recovering']).ffill()
        out['streak_extreme'] = out['streak'] > out['streak_threshold']
        ready = out[['drawdown', 'origin_drawdown', 'origin_threshold', 'streak_threshold']].notna().all(axis=1)
        out['entry'] = (
            ready
            & out['recovering']
            & (out['drawdown'] < 0)
            & (out['origin_drawdown'] < 0)
            & (out['origin_drawdown'] <= out['origin_threshold'])
            & out['streak_extreme']
            & ~out['streak_extreme'].shift(1, fill_value=False)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                exit_bar = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=25, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': False}, index=index)
    indicators.loc[index[[1, 3]], 'entry'] = True
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[2:20] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert (result['size'] > 0).all()
