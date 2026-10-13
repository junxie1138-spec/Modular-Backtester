from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    rank_window: int = 60
    depth_percentile: float = 0.75
    recovery_percentile: float = 0.80
    hold_bars: int = 2

    def __post_init__(self):
        if self.peak_window < 2 or self.rank_window < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.0 < self.depth_percentile <= 1.0:
            raise ValueError('Invalid depth percentile')
        if not 0.0 < self.recovery_percentile <= 1.0:
            raise ValueError('Invalid recovery percentile')
        if self.hold_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791326153'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(params.peak_window + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['peak'] = data['close'].rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - data['close'].div(out['peak'].where(out['peak'] > 0))
        out['depth_rank'] = out['drawdown'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['prior_drawdown'] = out['drawdown'].shift(1)
        out['prior_depth_rank'] = out['depth_rank'].shift(1)
        out['prior_close'] = data['close'].shift(1)
        out['downside_excursion'] = out['prior_close'] - data['low']
        out['recovered_distance'] = data['close'] - data['low']
        out['recovery_fraction'] = out['recovered_distance'].div(out['downside_excursion'].where(out['downside_excursion'] > 0))
        out['recovery_fraction'] = out['recovery_fraction'].where(out['downside_excursion'] > 0, 0.0)
        out.loc[out['prior_close'].isna(), 'recovery_fraction'] = np.nan
        out['recovery_fraction'] = out['recovery_fraction'].replace([np.inf, -np.inf], np.nan)
        out['recovery_rank'] = out['recovery_fraction'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['depth_agrees'] = (
            out['prior_depth_rank'].ge(params.depth_percentile)
            & out['prior_drawdown'].gt(0.0)
            & out['drawdown'].lt(out['prior_drawdown'])
        ).fillna(False)
        out['retracement_agrees'] = (
            out['recovery_rank'].ge(params.recovery_percentile)
            & out['recovery_fraction'].gt(1.0)
        ).fillna(False)
        out['entry'] = out['depth_agrees'] & out['retracement_agrees']
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        eligible = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entered = -1
        for i, ready in enumerate(eligible):
            if entered >= 0:
                if i - entered >= params.hold_bars:
                    entered = -1
                else:
                    signal[i] = 1
            elif ready:
                entered = i
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RecoveryParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    indicators['entry'] = [True] + [False] * 9
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RecoveryParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
