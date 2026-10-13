from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCouplingParams:
    flow_window: int = 4
    risk_window: int = 20
    entry_threshold: float = 0.35
    target_vol: float = 0.12
    min_size: float = 0.05
    max_size: float = 0.95

    def __post_init__(self):
        if self.flow_window < 3 or self.risk_window < 2:
            raise ValueError('Lookbacks must be at least three and two bars.')
        values = [self.entry_threshold, self.target_vol, self.min_size, self.max_size]
        if not np.isfinite(values).all():
            raise ValueError('Parameters must be finite.')
        if not 0 < self.entry_threshold < 1 or self.target_vol <= 0:
            raise ValueError('Invalid correlation threshold or volatility target.')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[RangeCouplingParams]):
    strategy_id = 'gen_local-w2_1791211656'

    @classmethod
    def params_type(cls):
        return RangeCouplingParams

    def warmup_bars(self, params):
        return int(max(params.risk_window + 1, params.flow_window + 2))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda s: s > 0)
        session_open = data['open'].where(lambda s: s > 0)
        components = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1)
        out['range_fraction'] = components.max(axis=1) / previous_close
        out['intraday_return'] = data['close'] / session_open - 1.0
        out['close_return'] = data['close'] / previous_close - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out['coupling'] = out['range_fraction'].rolling(
            params.flow_window, min_periods=params.flow_window
        ).corr(out['intraday_return']).clip(-1.0, 1.0)
        out['annual_vol'] = out['close_return'].rolling(
            params.risk_window, min_periods=params.risk_window
        ).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        score = indicators['coupling']
        ready = score.notna() & indicators['annual_vol'].notna()
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[ready & score.ge(params.entry_threshold)] = 1
        raw.loc[ready & score.le(-params.entry_threshold)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)), 0)
        position = 0
        positions = np.zeros(len(data), dtype=int)
        for i, (side, entry) in enumerate(zip(raw.to_numpy(), confirmed.to_numpy())):
            if position != 0 and side == -position:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        volatility = indicators['annual_vol'].clip(lower=1e-8)
        decision_size = (params.target_vol / volatility).clip(
            lower=params.min_size, upper=params.max_size
        ).fillna(params.min_size)
        df['size'] = decision_size.shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RangeCouplingParams()
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'coupling': [0, 0.8, 0.8, 0, -0.8, -0.8, -0.8, 0, 0.8, 0.8],
        'annual_vol': [0.24] * 10,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, -1, 0]
    assert np.allclose(result['size'].iloc[1:], 0.5)
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
