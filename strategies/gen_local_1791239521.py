from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_length: int = 3
    target_daily_vol: float = 0.006

    def __post_init__(self):
        if isinstance(self.streak_length, bool) or not isinstance(self.streak_length, int) or self.streak_length < 1:
            raise ValueError('streak_length must be a positive integer')
        if not np.isfinite(self.target_daily_vol) or self.target_daily_vol <= 0:
            raise ValueError('target_daily_vol must be finite and positive')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791239521'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(21, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        returns = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        down = returns.lt(0)
        streak = down.astype(int).groupby((~down).cumsum()).cumsum()
        volatility = returns.rolling(20, min_periods=20).std(ddof=0)
        decline = -returns.rolling(params.streak_length, min_periods=params.streak_length).sum()
        tension = decline.div(volatility.where(volatility.gt(0)) * np.sqrt(params.streak_length))
        entry_size = params.target_daily_vol / volatility.where(volatility.gt(0))
        eligible = streak.eq(params.streak_length) & tension.ge(1.0) & volatility.gt(0)
        return pd.DataFrame({
            'return': returns.fillna(0.0),
            'down': down,
            'streak': streak.astype(int),
            'volatility': volatility.fillna(0.0),
            'decline': decline.fillna(0.0),
            'tension': tension.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            'entry_size': entry_size.clip(lower=0.000001, upper=0.95).fillna(0.95),
            'eligible': eligible.fillna(False),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.95, dtype=float)
        eligible = indicators['eligible'].to_numpy(dtype=bool)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        remaining = 0
        held_size = 0.95
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                if remaining > 0:
                    signal[i] = 1
                    size[i] = held_size
                # Expiration always emits zero, even if another entry qualifies.
            elif eligible[i]:
                remaining = 2
                held_size = float(entry_sizes[i])
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'eligible': [True, False, True, True, False, True, True],
        'entry_size': [0.4, 0.8, 0.9, 0.6, 0.8, 0.9, 0.7],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert np.allclose(result.loc[result['signal'].eq(1), 'size'], [0.4, 0.4, 0.6, 0.6])
