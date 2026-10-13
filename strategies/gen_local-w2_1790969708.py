from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPersistenceParams:
    volume_window: int = 20
    wave_window: int = 8
    volume_multiple: float = 1.1
    reversal_threshold: float = 0.035

    def __post_init__(self):
        if self.volume_window < 1 or self.wave_window < 1:
            raise ValueError('Lookback windows must be positive.')
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 0:
            raise ValueError('volume_multiple must be finite and positive.')
        if not np.isfinite(self.reversal_threshold) or self.reversal_threshold <= 0:
            raise ValueError('reversal_threshold must be finite and positive.')


class GeneratedStrategy(BaseStrategy[GapPersistenceParams]):
    strategy_id = 'gen_local-w2_1790969708'

    @classmethod
    def params_type(cls):
        return GapPersistenceParams

    def warmup_bars(self, params: GapPersistenceParams) -> int:
        return int(params.volume_window + params.wave_window)

    def indicators(self, data: pd.DataFrame, params: GapPersistenceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['gap'] = data['open'] - out['previous_close']
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).replace(0.0, np.nan)
        out['volume_baseline'] = data['volume'].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1).replace(0.0, np.nan)
        out['relative_volume'] = data['volume'] / out['volume_baseline']
        out['surviving_gap'] = (
            (data['low'] - out['previous_close']).clip(lower=0.0)
            - (out['previous_close'] - data['high']).clip(lower=0.0)
        )
        valid = (
            np.isfinite(data).all(axis=1)
            & (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            & (data['volume'] >= 0)
            & (data['high'] >= data[['open', 'close', 'low']].max(axis=1))
            & (data['low'] <= data[['open', 'close', 'high']].min(axis=1))
            & np.isfinite(out['previous_close'])
            & (out['previous_close'] > 0)
            & np.isfinite(out['relative_volume'])
            & np.isfinite(out['true_range'])
            & (out['true_range'] > 0)
        )
        out['volume_confirmed'] = (
            valid & (out['relative_volume'] >= params.volume_multiple)
        )
        out['impulse'] = (
            out['surviving_gap'] / out['true_range']
            * out['relative_volume'].clip(upper=3.0)
        ).where(out['volume_confirmed'], 0.0).where(valid)
        out['gap_pressure'] = out['impulse'].rolling(
            params.wave_window, min_periods=params.wave_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPersistenceParams,
    ) -> SignalFrame:
        pressure = indicators['gap_pressure']
        finite = np.isfinite(pressure)
        entry = finite & (pressure >= params.reversal_threshold)
        exit_ = finite & (pressure <= -params.reversal_threshold)
        # Mirrored evidence changes the position; neutral evidence preserves it.
        decisions = pd.Series(
            np.select([entry, exit_], [1.0, 0.0], default=np.nan),
            index=data.index,
        ).ffill().fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapPersistenceParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {'gap_pressure': [np.nan, 0.05, 0.0, -0.05, np.nan, 0.05]},
        index=index,
    )
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=28)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) == 28


if __name__ == '__main__':
    _self_check()
