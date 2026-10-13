from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakRegimeParams:
    regime_window: int = 40
    streak_bars: int = 3
    regime_margin: float = 0.04
    hold_bars: int = 7
    position_size: float = 0.95

    def __post_init__(self):
        if self.regime_window < 5 or self.streak_bars < 2:
            raise ValueError('Require regime_window >= 5 and streak_bars >= 2')
        if not 5 <= self.hold_bars <= 10:
            raise ValueError('hold_bars must be between 5 and 10')
        if not 0 < self.regime_margin < 1:
            raise ValueError('regime_margin must be between 0 and 1')
        if not 0 < self.position_size <= 1:
            raise ValueError('position_size must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[StreakRegimeParams]):
    strategy_id = 'gen_local_1791243657'

    @classmethod
    def params_type(cls):
        return StreakRegimeParams

    def warmup_bars(self, params: StreakRegimeParams) -> int:
        return int(max(params.regime_window + 2, params.streak_bars + 2))

    def indicators(self, data: pd.DataFrame, params: StreakRegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['change'] = data['close'].diff()
        out['direction'] = np.sign(out['change'])
        up = out['direction'].eq(1)
        down = out['direction'].eq(-1)
        out['up_streak'] = up.groupby((~up).cumsum()).cumsum().astype(int)
        out['down_streak'] = down.groupby((~down).cumsum()).cumsum().astype(int)
        previous = out['direction'].shift(1)
        out['same_sign'] = out['direction'].eq(previous).astype(float).where(
            out['direction'].notna() & previous.notna()
        )
        window = params.regime_window
        out['repeat_rate'] = out['same_sign'].rolling(window, min_periods=window).mean()
        for name, sign in (('up_fraction', 1), ('down_fraction', -1), ('flat_fraction', 0)):
            out[name] = out['direction'].eq(sign).astype(float).where(
                out['direction'].notna()
            ).rolling(window + 1, min_periods=window + 1).mean()
        out['independent_repeat'] = (
            out['up_fraction'].pow(2)
            + out['down_fraction'].pow(2)
            + out['flat_fraction'].pow(2)
        )
        out['excess_repeat'] = out['repeat_rate'] - out['independent_repeat']
        ready = out['excess_repeat'].notna()
        out['continuation'] = (
            ready
            & out['excess_repeat'].ge(params.regime_margin)
            & out['up_streak'].ge(params.streak_bars)
        ).fillna(False)
        out['reversion'] = (
            ready
            & out['excess_repeat'].le(-params.regime_margin)
            & out['down_streak'].ge(params.streak_bars)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakRegimeParams,
    ) -> SignalFrame:
        continuation = indicators['continuation'].fillna(False).astype(bool)
        reversion = indicators['reversion'].fillna(False).astype(bool)
        confirmed = (
            (continuation & continuation.shift(1, fill_value=False))
            | (reversion & reversion.shift(1, fill_value=False))
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif confirmed[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': float(params.position_size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StreakRegimeParams(hold_bars=5)
    index = pd.date_range('2020-01-01', periods=20, freq='B')
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    candidates = pd.DataFrame(False, index=index, columns=['continuation', 'reversion'])
    candidates.loc[index[1:7], 'continuation'] = True
    candidates.loc[index[8], 'reversion'] = True
    candidates.loc[index[10:12], 'reversion'] = True
    result = strategy.generate_signals(data, candidates, None, params).data
    assert result['signal'].tolist() == [0] * 3 + [1] * 5 + [0] * 4 + [1] * 5 + [0] * 3
    assert result['size'].gt(0).all()
    assert not strategy.indicators(data, params)[['continuation', 'reversion']].to_numpy().any()
