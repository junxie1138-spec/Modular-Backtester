from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EncroachmentParams:
    channel_bars: int = 20
    pressure_bars: int = 5
    rank_bars: int = 60
    entry_percentile: float = 0.80
    hold_bars: int = 8
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[EncroachmentParams]):
    strategy_id = 'gen_local-w2_1791283854'

    @classmethod
    def params_type(cls) -> type[EncroachmentParams]:
        return EncroachmentParams

    @staticmethod
    def warmup_bars(params: EncroachmentParams) -> int:
        if min(params.channel_bars, params.pressure_bars, params.rank_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError('Entry percentile must be between zero and one.')
        if not 0.0 < params.minimum_size <= 1.0:
            raise ValueError('Minimum size must be positive and at most one.')
        return int(params.channel_bars + params.pressure_bars + params.rank_bars)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EncroachmentParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['floor'] = data['low'].rolling(params.channel_bars).min()
        out['ceiling'] = data['high'].rolling(params.channel_bars).max()
        out['width'] = (out['ceiling'] - out['floor']).where(lambda x: x > 0.0)
        out['floor_advance'] = out['floor'].diff(params.pressure_bars)
        out['ceiling_advance'] = out['ceiling'].diff(params.pressure_bars)
        out['encroachment'] = (
            (out['floor_advance'] - out['ceiling_advance']) / out['width']
        ).replace([np.inf, -np.inf], np.nan)
        out['pressure_rank'] = out['encroachment'].rolling(params.rank_bars).rank(pct=True)
        out['prior_rank'] = out['pressure_rank'].shift(1)
        out['prior_ceiling'] = out['ceiling'].shift(1)
        out['prior_encroachment'] = out['encroachment'].shift(1)
        out['entry'] = (
            (data['close'] > out['prior_ceiling'])
            & (out['prior_encroachment'] > 0.0)
            & (out['prior_rank'] >= params.entry_percentile)
        ).fillna(False)
        out['strength'] = (
            (out['prior_rank'] - params.entry_percentile)
            / (1.0 - params.entry_percentile)
        ).clip(0.0, 1.0).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EncroachmentParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        exit_at = -1
        entry_size = params.minimum_size
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif entries[i]:
                exit_at = i + params.hold_bars
                entry_size = params.minimum_size + (1.0 - params.minimum_size) * strengths[i]
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = EncroachmentParams(hold_bars=3)
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    data = pd.DataFrame(1.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = pd.DataFrame({'entry': True, 'strength': np.arange(12) / 11.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    assert (result['size'] > 0.0).all()
    assert result['size'].iloc[1:4].nunique() == 1
    assert result['size'].iloc[5:8].nunique() == 1
    assert result['size'].iloc[5] > result['size'].iloc[1]
    flat = GeneratedStrategy.indicators(data, params)
    assert not flat['entry'].any()
    assert GeneratedStrategy.warmup_bars(params) == 85


if __name__ == '__main__':
    _self_check()
