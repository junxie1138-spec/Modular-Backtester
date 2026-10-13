from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapQueueParams:
    queue_window: int = 10
    vol_window: int = 20
    backlog_threshold: float = 1.5
    gap_threshold: float = 0.3
    gap_ceiling: float = 1.5
    target_vol: float = 0.12
    holding_bars: int = 4

    def __post_init__(self):
        if self.queue_window < 2 or self.vol_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 3 <= self.holding_bars <= 5:
            raise ValueError('Holding period must be three to five bars.')
        values = [self.backlog_threshold, self.gap_threshold,
                  self.gap_ceiling, self.target_vol]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds and target volatility must be finite and positive.')
        if self.gap_threshold >= self.gap_ceiling:
            raise ValueError('Gap threshold must be below the ceiling.')


class GeneratedStrategy(BaseStrategy[GapQueueParams]):
    strategy_id = 'gen_local-w2_1791234292'

    @classmethod
    def params_type(cls):
        return GapQueueParams

    def warmup_bars(self, params: GapQueueParams) -> int:
        return int(max(params.queue_window, params.vol_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapQueueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data['close'].shift(1).replace(0.0, np.nan)
        out['gap'] = data['open'] / prior_close - 1.0
        out['return'] = data['close'].pct_change(fill_method=None)
        out['gap_vol'] = out['gap'].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0).shift(1).replace(0.0, np.nan)
        out['annual_vol'] = out['return'].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0).replace(0.0, np.nan) * np.sqrt(252.0)
        out['down_load'] = (-out['gap']).clip(lower=0.0).rolling(
            params.queue_window, min_periods=params.queue_window
        ).sum().shift(1)
        out['up_load'] = out['gap'].clip(lower=0.0).rolling(
            params.queue_window, min_periods=params.queue_window
        ).sum().shift(1)
        out['backlog'] = (out['down_load'] - out['up_load']).clip(lower=0.0)
        out['backlog_z'] = out['backlog'] / out['gap_vol']
        out['gap_z'] = out['gap'] / out['gap_vol']
        out['intraday_return'] = data['close'] / data['open'].replace(0.0, np.nan) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out['eligible'] = (
            out['backlog_z'].ge(params.backlog_threshold)
            & out['gap_z'].ge(params.gap_threshold)
            & out['gap_z'].le(params.gap_ceiling)
            & out['intraday_return'].ge(0.0)
            & out['annual_vol'].gt(0.0)
        ).fillna(False)
        out['conviction'] = (
            out['gap_z'].clip(lower=0.0, upper=params.gap_ceiling)
            / params.gap_ceiling
            * out['backlog_z'] / (1.0 + out['backlog_z'])
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        out['entry_size'] = (
            params.target_vol / out['annual_vol'] * out['conviction']
        ).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapQueueParams) -> SignalFrame:
        eligible = indicators['eligible'].fillna(False).to_numpy(dtype=bool)
        entry_size = indicators['entry_size'].fillna(0.05).clip(0.05, 0.95).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        exit_at = -1
        locked_size = 1.0
        for i in range(len(data)):
            if i < exit_at:
                signal[i] = 1
                size[i] = locked_size
            elif i > exit_at and eligible[i]:
                exit_at = i + params.holding_bars
                locked_size = entry_size[i]
                signal[i] = 1
                size[i] = locked_size
            # At exit_at, force a flat bar even if another entry qualifies.
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GapQueueParams()
    index = pd.date_range('2020-01-01', periods=11)
    data = pd.DataFrame(index=index)
    entry_sizes = np.linspace(0.1, 0.9, len(index))
    indicators = pd.DataFrame({'eligible': True, 'entry_size': entry_sizes}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert result['size'].iloc[1:5].eq(entry_sizes[0]).all()
    assert result['size'].iloc[6:10].eq(entry_sizes[5]).all()
    assert result['size'].gt(0.0).all()
