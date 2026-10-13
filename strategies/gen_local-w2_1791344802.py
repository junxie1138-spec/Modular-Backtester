from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumImpactParams:
    momentum_bars: int = 3
    rank_window: int = 60
    impact_window: int = 12
    rank_threshold: float = 0.85
    impact_ratio: float = 1.2
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.momentum_bars, self.rank_window, self.impact_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between 3 and 5.')
        if not np.isfinite(self.rank_threshold) or not 0 < self.rank_threshold <= 1:
            raise ValueError('rank_threshold must be in (0, 1].')
        if not np.isfinite(self.impact_ratio) or self.impact_ratio <= 1:
            raise ValueError('impact_ratio must exceed 1.')


def _fixed_hold(eligible, hold_bars):
    signal = np.zeros(len(eligible), dtype=np.int64)
    exit_bar = -1
    for i, enter in enumerate(eligible):
        if exit_bar >= 0:
            if i < exit_bar:
                signal[i] = 1
            else:
                exit_bar = -1
        elif enter:
            signal[i] = 1
            exit_bar = i + hold_bars
    return signal


class GeneratedStrategy(BaseStrategy[MomentumImpactParams]):
    strategy_id = 'gen_local-w2_1791344802'

    @classmethod
    def params_type(cls):
        return MomentumImpactParams

    def warmup_bars(self, params):
        return int(max(params.rank_window + params.momentum_bars, params.impact_window))

    def indicators(self, data, params):
        result = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0).replace([np.inf, -np.inf], np.nan)
        opening = data['open'].where(data['open'] > 0).replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].where(data['volume'] > 0).replace([np.inf, -np.inf], np.nan)
        result['momentum'] = close.pct_change(params.momentum_bars, fill_method=None)
        result['momentum_rank'] = result['momentum'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='average', pct=True)
        result['body_return'] = ((close - opening) / opening).where(volume.notna())
        body = result['body_return']
        window = params.impact_window
        result['buy_move'] = body.clip(lower=0).rolling(window, min_periods=window).sum()
        result['sell_move'] = (-body.clip(upper=0)).rolling(window, min_periods=window).sum()
        result['buy_volume'] = volume.where(body > 0, 0.0).where(body.notna()).rolling(
            window, min_periods=window
        ).sum()
        result['sell_volume'] = volume.where(body < 0, 0.0).where(body.notna()).rolling(
            window, min_periods=window
        ).sum()
        result['buy_impact'] = result['buy_move'] / result['buy_volume'].replace(0, np.nan)
        result['sell_impact'] = result['sell_move'] / result['sell_volume'].replace(0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        momentum_agrees = (
            (indicators['momentum'] > 0)
            & (indicators['momentum_rank'] >= params.rank_threshold)
        )
        impact_agrees = (
            (indicators['sell_impact'] > 0)
            & (indicators['buy_impact'] > params.impact_ratio * indicators['sell_impact'])
        )
        eligible = (momentum_agrees & impact_agrees).fillna(False).to_numpy(dtype=bool)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_hold(eligible, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    for hold in (3, 4, 5):
        raw = _fixed_hold(np.ones(20, dtype=bool), hold)
        shifted = pd.Series(raw).shift(1).fillna(0).astype(int)
        assert shifted.iloc[0] == 0
        assert (shifted.iloc[1:hold + 1] == 1).all()
        assert shifted.iloc[hold + 1] == 0
        assert shifted.iloc[hold + 2] == 1
    assert not _fixed_hold(np.zeros(20, dtype=bool), 4).any()
