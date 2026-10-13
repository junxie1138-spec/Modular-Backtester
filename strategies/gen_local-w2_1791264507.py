from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakBreakoutParams:
    breakout_window: int = 20
    min_streak: int = 3
    streak_cap: int = 8
    hold_bars: int = 2
    min_size: float = 0.35
    max_size: float = 0.95

    def __post_init__(self):
        if self.breakout_window < 2:
            raise ValueError('breakout_window must be at least 2')
        if not 1 <= self.min_streak <= self.streak_cap:
            raise ValueError('Require 1 <= min_streak <= streak_cap')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Require 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[StreakBreakoutParams]):
    strategy_id = 'gen_local-w2_1791264507'

    @classmethod
    def params_type(cls):
        return StreakBreakoutParams

    def warmup_bars(self, params: StreakBreakoutParams) -> int:
        return int(max(params.breakout_window, params.streak_cap) + 1)

    def indicators(self, data: pd.DataFrame, params: StreakBreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        low = data['low'].replace([np.inf, -np.inf], np.nan)
        high = data['high'].replace([np.inf, -np.inf], np.nan)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        out['higher_low'] = low.gt(low.shift(1)) & low.gt(0)
        # Capping makes streak dependence finite: streak_cap + 1 low observations.
        out['streak'] = (
            out['higher_low'].astype(int)
            .groupby((~out['higher_low']).cumsum())
            .cumsum()
            .clip(upper=params.streak_cap)
        )
        out['ceiling'] = high.shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        out['breakout'] = close.gt(out['ceiling']) & out['ceiling'].gt(0)
        out['conviction'] = (
            (out['streak'] - params.min_streak + 1)
            / (params.streak_cap - params.min_streak + 1)
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        out['entry_size'] = (
            params.min_size + (params.max_size - params.min_size) * out['conviction']
        ).astype(float)
        out['entry'] = (
            out['breakout']
            & out['streak'].ge(params.min_streak)
            & low.notna()
            & high.notna()
            & close.notna()
            & close.gt(0)
        )
        out.loc[out.index[:self.warmup_bars(params) - 1], 'entry'] = False
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakBreakoutParams,
    ) -> SignalFrame:
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        entry_size = indicators['entry_size'].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(params.min_size).clip(
            lower=params.min_size, upper=params.max_size
        ).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        frozen_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0 and i >= exit_bar:
                exit_bar = -1
                # Always emit a flat bar, even if another entry qualifies.
                continue
            if exit_bar < 0 and entry[i]:
                exit_bar = i + params.hold_bars
                frozen_size = float(entry_size[i])
            if exit_bar >= 0:
                signal[i] = 1
                size[i] = frozen_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        # Entry and exit decisions both execute on the following bar.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=9, freq='D')
    low = np.array([5., 6., 7., 8., 7., 8., np.nan, 9., 10.])
    data = pd.DataFrame({
        'open': low + 0.5, 'high': low + 1.0,
        'low': low, 'close': low + 0.8, 'volume': 1000.0,
    }, index=index)
    derived = strategy.indicators(data, StreakBreakoutParams())
    assert derived['streak'].tolist() == [0, 1, 2, 3, 0, 1, 0, 0, 1]
    assert not derived['entry'].any()
    test_data = data.iloc[:7]
    test_indicators = pd.DataFrame({
        'entry': True, 'entry_size': np.linspace(0.35, 0.95, 7),
    }, index=test_data.index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0]),
        (2, [0, 1, 1, 0, 1, 1, 0]),
    ):
        result = strategy.generate_signals(
            test_data, test_indicators, None, StreakBreakoutParams(hold_bars=hold)
        ).data
        assert result['signal'].tolist() == expected
        assert result['size'].gt(0).all()
        if hold == 2:
            assert result['size'].iloc[1] == result['size'].iloc[2] == 0.35


if __name__ == '__main__':
    _self_check()
