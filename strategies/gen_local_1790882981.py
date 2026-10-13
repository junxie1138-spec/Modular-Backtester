from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    loss_window: int = 10
    rank_window: int = 60
    tail_percentile: float = 0.20
    hold_bars: int = 4

    def __post_init__(self):
        if self.loss_window < 2 or self.rank_window < 2:
            raise ValueError('Lookback windows must be at least two bars')
        if not 0.0 < self.tail_percentile < 0.5:
            raise ValueError('tail_percentile must be between zero and 0.5')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790882981'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.loss_window + params.rank_window,
                       3 + params.rank_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return'] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out['downside_energy'] = out['return'].clip(upper=0).pow(2)
        window = out['downside_energy'].rolling(
            params.loss_window, min_periods=params.loss_window)
        out['loss_sum'] = window.sum()
        out['loss_max'] = window.max()
        out['concentration'] = out['loss_max'].div(
            out['loss_sum'].where(out['loss_sum'] > 0))
        out.loc[out['loss_sum'].eq(0), 'concentration'] = 0.0
        out['concentration_rank'] = out['concentration'].rolling(
            params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['move'] = close.pct_change(3, fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out['move_rank'] = out['move'].rolling(
            params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = indicators[['ma200', 'concentration_rank', 'move',
                            'move_rank']].notna().all(axis=1)
        bull = data['close'].gt(indicators['ma200'])
        reversal = (
            indicators['concentration_rank'].ge(0.80)
            & indicators['move_rank'].le(params.tail_percentile)
            & indicators['move'].lt(0)
        )
        continuation = (
            indicators['concentration_rank'].le(0.50)
            & indicators['move_rank'].ge(1.0 - params.tail_percentile)
            & indicators['move'].gt(0)
        )
        entries = (ready & bull & (reversal | continuation)).fillna(False).to_numpy()
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
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame({'close': 100.0}, index=index)
    indicators = pd.DataFrame({
        'ma200': 90.0, 'concentration_rank': 1.0,
        'move': -0.03, 'move_rank': 0.05,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    indicators.loc[:, 'ma200'] = np.nan
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, StrategyParams()).data
    assert result['signal'].eq(0).all()
