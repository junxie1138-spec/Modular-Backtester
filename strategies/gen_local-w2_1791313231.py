from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    rank_window: int = 60
    volume_window: int = 20
    queue_window: int = 3
    rank_threshold: float = 0.85
    profit_target: float = 0.008
    max_hold: int = 2
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = 'gen_local-w2_1791313231'

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    @staticmethod
    def warmup_bars(params: GapPressureParams) -> int:
        if min(params.rank_window, params.volume_window, params.queue_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.rank_threshold < 1.0:
            raise ValueError('rank_threshold must be between zero and one.')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite.')
        if params.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions.')
        if not 0.0 < params.min_size <= 1.0:
            raise ValueError('min_size must be in (0, 1].')
        return int(params.volume_window + params.queue_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = data['open'] / out['previous_close'] - 1.0
        out['volume_reference'] = data['volume'].rolling(params.volume_window).median().shift(1)
        out['relative_volume'] = data['volume'].where(data['volume'] > 0) / out['volume_reference'].where(out['volume_reference'] > 0)
        out['gap_load'] = out['gap'] / out['relative_volume'].clip(lower=0.25)
        out['queue'] = out['gap_load'].rolling(params.queue_window).sum()
        out['gap_direction_sum'] = np.sign(out['gap']).rolling(params.queue_window).sum()
        out['pressure_rank'] = out['queue'].abs().rolling(params.rank_window).rank(pct=True)
        out['intraday_return'] = data['close'] / data['open'].where(data['open'] > 0) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        direction = np.sign(out['queue'])
        eligible = (
            out['pressure_rank'].ge(params.rank_threshold)
            & out['gap_direction_sum'].abs().eq(params.queue_window)
            & (direction * out['intraday_return']).lt(0)
            & data['close'].gt(0)
        )
        out['entry'] = (-direction).where(eligible, 0).fillna(0).astype(int)
        out['conviction'] = (
            params.min_size
            + (1.0 - params.min_size)
            * ((out['pressure_rank'] - params.rank_threshold) / (1.0 - params.rank_threshold)).clip(0, 1)
        ).fillna(params.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        n = len(data)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        conviction = indicators['conviction'].fillna(params.min_size).to_numpy(dtype=float)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        trade_size = params.min_size

        for i in range(n):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    continue
                # Close-confirmed exits execute at the following open.
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    continue
            elif entries[i] and i + 1 < n:
                position = int(entries[i])
                entry_bar = i + 1
                trade_size = float(np.clip(conviction[i], params.min_size, 1.0))

            signals[i] = position
            if position:
                sizes[i] = trade_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = GapPressureParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': np.full(7, 100.0),
        'high': np.full(7, 102.0),
        'low': np.full(7, 98.0),
        'close': [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'volume': np.full(7, 1000.0),
    }, index=index)
    indicators = pd.DataFrame({
        'entry': [1, 0, -1, 0, 0, 0, 0],
        'conviction': [0.8, 0.25, 0.6, 0.25, 0.25, 0.25, 0.25],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, -1, -1, 0, 0]
    assert np.allclose(result['size'].iloc[[1, 3, 4]], [0.8, 0.6, 0.6])
    data['close'] = [100.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    indicators['entry'] = [-1, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
