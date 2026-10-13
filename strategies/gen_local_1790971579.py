from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 63
    recovery_window: int = 15
    volatility_window: int = 20
    minimum_depth: float = 0.8
    recovery_fraction: float = 0.25
    target_volatility: float = 0.12
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.peak_window, self.recovery_window, self.volatility_window) < 2:
            raise ValueError('Indicator windows must be at least two bars')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        if not np.isfinite(self.minimum_depth) or self.minimum_depth <= 0:
            raise ValueError('minimum_depth must be finite and positive')
        if not 0 < self.recovery_fraction < 1:
            raise ValueError('recovery_fraction must be between zero and one')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError('target_volatility must be finite and positive')


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = 'gen_local_1790971579'

    @classmethod
    def params_type(cls):
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        return int(max(params.peak_window + params.recovery_window,
                       params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['peak'] = close.rolling(params.peak_window).max()
        out['depth'] = (1.0 - close / out['peak']).clip(lower=0.0)
        out['daily_volatility'] = out['return'].rolling(params.volatility_window).std(ddof=0)
        out['annual_volatility'] = out['daily_volatility'] * np.sqrt(252.0)
        out['maximum_depth'] = out['depth'].rolling(params.recovery_window).max()
        out['recovery'] = 1.0 - out['depth'] / out['maximum_depth'].where(out['maximum_depth'] > 0)
        out['previous_recovery'] = out['recovery'].shift(1)
        out['normalized_depth'] = out['depth'] / (
            out['daily_volatility'].where(out['daily_volatility'] > 0)
            * np.sqrt(float(params.volatility_window))
        )
        out['strength'] = (out['normalized_depth'] / (2.0 * params.minimum_depth)).clip(0.25, 1.0)
        out['entry_size'] = (
            params.target_volatility
            / out['annual_volatility'].where(out['annual_volatility'] > 0)
            * out['strength']
        ).clip(0.05, 0.95)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['eligible'] = (
            (out['normalized_depth'] >= params.minimum_depth)
            & (out['previous_recovery'] < params.recovery_fraction)
            & (out['recovery'] >= params.recovery_fraction)
            & out['entry_size'].notna()
        ).fillna(False)
        out['entry_size'] = out['entry_size'].fillna(0.95)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        count = len(data)
        signal = np.zeros(count, dtype=int)
        size = np.full(count, 0.95, dtype=float)
        eligible = indicators['eligible'].fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        entry_bar = -1
        held_size = 0.95
        for i in range(count):
            if entry_bar >= 0:
                if i - entry_bar == params.holding_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
                    size[i] = held_size
            elif eligible[i] and np.isfinite(entry_sizes[i]) and entry_sizes[i] > 0:
                entry_bar = i
                held_size = float(entry_sizes[i])
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        'eligible': [True, False, False, True, True, False, False, False, False],
        'entry_size': [0.4] + [0.8] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        data, derived, None, DrawdownParams(holding_bars=3)
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0]
    assert np.allclose(result['size'].iloc[1:4], 0.4)
    assert np.allclose(result['size'].iloc[5:8], 0.8)
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
