from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    hub_degree: int = 4
    hold_bars: int = 7

    def __post_init__(self):
        if not 3 <= self.hub_degree <= 20:
            raise ValueError('hub_degree must be between 3 and 20')
        if not 5 <= self.hold_bars <= 10:
            raise ValueError('hold_bars must be between 5 and 10')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791275634'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # Twenty backward edges need 21 returns; prior degree needs one more bar.
        return 23

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_return'] = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        out['downside_shock'] = -out['log_return']
        shock = out['downside_shock']
        current = shock.to_numpy()
        blocker = np.full(len(data), -np.inf)
        degree = np.zeros(len(data), dtype=int)
        # Horizontal visibility: both endpoints exceed every intervening node.
        for lag in range(1, 21):
            past = shock.shift(lag).to_numpy()
            visible = np.isfinite(current) & np.isfinite(past) & (np.minimum(current, past) > blocker)
            degree += visible
            blocker = np.maximum(blocker, past)
        complete = shock.rolling(21, min_periods=21).count().eq(21)
        out['degree'] = pd.Series(degree, index=data.index, dtype=float).where(complete)
        out['previous_degree'] = out['degree'].shift(1)
        day = data.index.day
        out['seasonal_window'] = (day >= 25) | (day <= 3)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = (
            indicators['seasonal_window']
            & indicators['previous_degree'].ge(params.hub_degree)
            & indicators['degree'].le(2)
            & indicators['log_return'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                    continue
                signal[i] = 1
            elif eligible[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Continuous eligibility must still produce an exit after exactly N bars.
    index = pd.bdate_range('2020-01-01', periods=40)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'seasonal_window': True,
        'previous_degree': 4.0,
        'degree': 1.0,
        'log_return': 0.01,
    }, index=index)
    strategy = GeneratedStrategy()
    for hold in (5, 7, 10):
        result = strategy.generate_signals(data, indicators, None, Params(hold_bars=hold)).data
        assert result['signal'].iloc[:hold + 3].tolist() == [0] + [1] * hold + [0, 1]
        assert result['size'].gt(0).all()
