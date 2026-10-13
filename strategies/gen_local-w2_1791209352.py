from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConcentrationParams:
    flow_window: int = 5
    rank_window: int = 63
    tail_fraction: float = 0.15
    hold_bars: int = 8
    min_size: float = 0.35
    max_size: float = 0.95

    def __post_init__(self):
        if self.flow_window < 2 or self.rank_window < 3 or self.hold_bars < 1:
            raise ValueError('Invalid lookback or holding period')
        if not 0.0 < self.tail_fraction < 0.5:
            raise ValueError('tail_fraction must lie between zero and one half')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[ConcentrationParams]):
    strategy_id = 'gen_local-w2_1791209352'

    @classmethod
    def params_type(cls):
        return ConcentrationParams

    def warmup_bars(self, params: ConcentrationParams) -> int:
        return int(params.flow_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: ConcentrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['daily_return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['direction'] = np.sign(out['daily_return'])
        out['valid_volume'] = data['volume'].where(np.isfinite(data['volume']) & data['volume'].ge(0))
        out['signed_volume'] = out['direction'] * out['valid_volume']
        out['volume_sum'] = out['valid_volume'].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out['signed_volume_sum'] = out['signed_volume'].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out['volume_imbalance'] = out['signed_volume_sum'] / out['volume_sum'].replace(0.0, np.nan)
        out['session_imbalance'] = out['direction'].rolling(params.flow_window, min_periods=params.flow_window).mean()
        out['concentration'] = (out['volume_imbalance'] - out['session_imbalance']).replace([np.inf, -np.inf], np.nan)
        # Native rolling percentile rank; ties receive their average rank.
        out['concentration_rank'] = out['concentration'].rolling(params.rank_window, min_periods=params.rank_window).rank(method='average', pct=True)
        out['net_return'] = data['close'].pct_change(periods=params.flow_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ConcentrationParams,
    ) -> SignalFrame:
        rank = indicators['concentration_rank']
        displacement = indicators['net_return']
        valid = np.isfinite(rank) & np.isfinite(displacement)
        long_entry = valid & rank.le(params.tail_fraction) & displacement.lt(0.0)
        short_entry = valid & rank.ge(1.0 - params.tail_fraction) & displacement.gt(0.0)
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        extremity = np.where(
            candidates == 1,
            (params.tail_fraction - rank.to_numpy()) / params.tail_fraction,
            (rank.to_numpy() - (1.0 - params.tail_fraction)) / params.tail_fraction,
        )
        conviction = np.clip(np.nan_to_num(extremity, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        entry_sizes = params.min_size + (params.max_size - params.min_size) * conviction

        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        exit_bar = -1
        frozen_size = params.min_size
        for i in range(len(data)):
            if side != 0 and i >= exit_bar:
                # Emit a flat bar even if another entry qualifies today.
                side = 0
                continue
            if side == 0 and candidates[i] != 0:
                side = int(candidates[i])
                exit_bar = i + params.hold_bars
                frozen_size = float(entry_sizes[i])
            if side != 0:
                signal[i] = side
                sizes[i] = frozen_size

        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = ConcentrationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'concentration_rank': 0.99, 'net_return': 0.01}, index=index)
    indicators.iloc[0] = [0.01, -0.01]
    indicators.iloc[2, 0] = 0.90
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:9].eq(1).all()
    assert result['signal'].iloc[9] == 0
    assert result['signal'].iloc[10:18].eq(-1).all()
    assert result['signal'].iloc[18] == 0
    assert result['size'].iloc[1:9].eq(result['size'].iloc[1]).all()
    assert result['size'].between(params.min_size, params.max_size).all()
    indicators['concentration_rank'] = np.nan
    assert strategy.generate_signals(data, indicators, ctx, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
