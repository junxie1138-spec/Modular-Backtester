from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 3
    atr_multiple: float = 2.0

    def __post_init__(self):
        if isinstance(self.streak_length, bool) or not isinstance(self.streak_length, int) or self.streak_length < 2:
            raise ValueError('streak_length must be an integer >= 2')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive')


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = 'gen_local-w2_1791325618'

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(65, params.streak_length + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['lag_return'] = out['return'].shift(1)
        out['autocorrelation'] = out['return'].rolling(63, min_periods=63).corr(out['lag_return']).replace([np.inf, -np.inf], np.nan)
        down = out['return'].lt(0)
        out['down_streak'] = down.astype(int).groupby((~down).cumsum()).cumsum()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = (
            indicators['down_streak'].eq(params.streak_length)
            & indicators['autocorrelation'].lt(0)
            & indicators['atr'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                # This is a held session: entry fills at its open.
                held += 1
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.atr_multiple * atr[i])
                    if close[i] <= stop or held >= 4:
                        active = False
                signals[i] = int(active)
                # An exit decision cannot also reopen the position.
                continue

            if valid and entry[i]:
                active = True
                held = 0
                # Exclude the decision bar from the in-trade high.
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StreakParams(streak_length=3, atr_multiple=2.0)
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    data = pd.DataFrame({'close': [100.0, 101.0, 104.0, 100.0, 100.0, 100.0]}, index=index)
    derived = pd.DataFrame({
        'down_streak': [3, 0, 0, 0, 0, 0],
        'autocorrelation': [-0.3] * 6,
        'atr': [1.0] * 6,
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    data['close'] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    derived['autocorrelation'] = np.nan
    assert strategy.generate_signals(data, derived, None, params).data['signal'].eq(0).all()
