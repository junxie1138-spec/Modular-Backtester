from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DistributionShiftParams:
    rank_window: int = 63
    baseline_window: int = 20
    recent_window: int = 5
    shock_window: int = 126
    shock_threshold: float = 0.90
    minimum_displacement: float = 0.15
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[DistributionShiftParams]):
    strategy_id = 'gen_local-w2_1791345958'

    @classmethod
    def params_type(cls) -> type[DistributionShiftParams]:
        return DistributionShiftParams

    def warmup_bars(self, params: DistributionShiftParams) -> int:
        return int(params.rank_window + params.baseline_window + params.recent_window + params.shock_window + 2)

    def indicators(self, data: pd.DataFrame, params: DistributionShiftParams) -> pd.DataFrame:
        if min(params.rank_window, params.baseline_window, params.recent_window, params.shock_window, params.hold_bars) < 1:
            raise ValueError('Window lengths and hold_bars must be positive.')
        if not 0.0 < params.shock_threshold < 1.0:
            raise ValueError('shock_threshold must lie strictly between zero and one.')
        if not 0.0 < params.minimum_displacement < 1.0:
            raise ValueError('minimum_displacement must lie strictly between zero and one.')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_rank'] = out['return'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['recent_rank'] = out['return_rank'].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['baseline_rank'] = out['return_rank'].shift(params.recent_window).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out['displacement'] = out['recent_rank'] - out['baseline_rank']
        out['shock_rank'] = out['displacement'].abs().rolling(params.shock_window, min_periods=params.shock_window).rank(pct=True)
        out['eligible'] = (
            out['shock_rank'].ge(params.shock_threshold)
            & out['displacement'].abs().ge(params.minimum_displacement)
            & out['baseline_rank'].between(0.35, 0.65)
            & (
                (out['displacement'].gt(0.0) & out['recent_rank'].ge(0.65))
                | (out['displacement'].lt(0.0) & out['recent_rank'].le(0.35))
            )
        ).fillna(False)
        out['entry_event'] = out['eligible'] & ~out['eligible'].shift(1, fill_value=False)
        out['strength'] = (
            0.5 * ((out['shock_rank'] - params.shock_threshold) / (1.0 - params.shock_threshold)).clip(0.0, 1.0)
            + 0.5 * (out['displacement'].abs() / 0.5).clip(0.0, 1.0)
        ).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DistributionShiftParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')
        events = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        displacement = indicators['displacement'].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = 0
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if active and i >= exit_bar:
                active = 0
            elif not active and events[i] and displacement[i] != 0.0:
                active = 1 if displacement[i] > 0.0 else -1
                exit_bar = i + params.hold_bars
                entry_size = 0.25 + 0.70 * strength[i]
            signals[i] = active
            sizes[i] = entry_size if active else 1.0
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = DistributionShiftParams()
    index = pd.date_range('2020-01-01', periods=42, freq='B')
    data = pd.DataFrame(index=index)
    fixture = pd.DataFrame({'entry_event': False, 'displacement': 0.3, 'strength': 0.5}, index=index)
    fixture.loc[index[[0, 3, 20]], 'entry_event'] = True
    fixture.loc[index[3]:, 'displacement'] = -0.3
    fixture.loc[index[3]:, 'strength'] = 1.0
    result = GeneratedStrategy().generate_signals(data, fixture, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:19].eq(1).all()
    assert result['signal'].iloc[19:21].eq(0).all()
    assert result['signal'].iloc[21:39].eq(-1).all()
    assert result['signal'].iloc[39:].eq(0).all()
    assert np.allclose(result['size'].iloc[1:19], 0.60)
    assert np.allclose(result['size'].iloc[21:39], 0.95)
    assert result['size'].gt(0.0).all()
