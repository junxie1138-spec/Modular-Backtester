from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_bars: int = 20
    recent_bars: int = 2
    prior_bars: int = 6
    z_threshold: float = 1.5
    pressure_threshold: float = 0.15
    hold_bars: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local-w2_1791289028'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if params.ma_bars < 2 or params.recent_bars < 1 or params.prior_bars < 1:
            raise ValueError('Lookbacks must be positive; ma_bars must be at least two.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError('z_threshold must be finite and positive.')
        if not np.isfinite(params.pressure_threshold) or not 0 < params.pressure_threshold < 1:
            raise ValueError('pressure_threshold must be between zero and one.')
        if not np.isfinite(params.size) or not 0 < params.size <= 1:
            raise ValueError('size must be between zero and one, inclusive of one.')
        return int(max(params.ma_bars + 1, params.recent_bars + params.prior_bars))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['ma'] = data['close'].rolling(params.ma_bars).mean()
        out['std'] = data['close'].rolling(params.ma_bars).std(ddof=0)
        out['z'] = (data['close'] - out['ma']) / out['std'].where(out['std'] > 0)
        out['range'] = data['high'] - data['low']
        out['body_pressure'] = (data['close'] - data['open']) / out['range'].where(out['range'] > 0)
        out['pressure_recent'] = out['body_pressure'].rolling(params.recent_bars).mean()
        out['pressure_prior'] = out['body_pressure'].shift(params.recent_bars).rolling(params.prior_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        z = indicators['z']
        distance_recovery = (
            (z.shift(1) <= -params.z_threshold)
            & (z > -params.z_threshold)
            & (z < 0)
        )
        pressure_changepoint = (
            (indicators['pressure_prior'] < -params.pressure_threshold)
            & (indicators['pressure_recent'] > params.pressure_threshold)
        )
        entry = (distance_recovery & pressure_changepoint).fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(entry):
            if remaining > 0:
                remaining -= 1
            elif enter:
                remaining = params.hold_bars
            # The expiry bar stays flat even if a new entry condition occurs.
            raw[i] = int(remaining > 0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10, freq='D')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'z': [-2.0, -1.0] * 5,
        'pressure_prior': [-0.5] * 10,
        'pressure_recent': [0.5] * 10,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=21)
    for hold, expected in (
        (2, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
        (1, [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
    ):
        result = strategy.generate_signals(data, indicators, ctx, GeneratedParams(hold_bars=hold))
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0).all()
    indicators['pressure_recent'] = -0.5
    assert not strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data['signal'].any()
    indicators['pressure_recent'] = np.nan
    assert not strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data['signal'].any()


if __name__ == '__main__':
    _self_check()
