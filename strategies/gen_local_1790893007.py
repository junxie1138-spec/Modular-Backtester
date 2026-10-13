from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    gap_sigma: float = 0.8
    profit_target: float = 0.006

    def __post_init__(self):
        if not np.isfinite(self.gap_sigma) or self.gap_sigma <= 0:
            raise ValueError('gap_sigma must be finite and positive')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = 'gen_local_1790893007'

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return 53

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        previous = close.shift(1)
        weights = np.ones(32)
        for k in range(1, 32):
            weights[k] = -weights[k - 1] * (0.45 - k + 1) / k
        # ponytail: truncate fractional memory at 32 bars; extend if longer memory is required.
        fractional = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(k)
        innovation = fractional.diff() / previous
        noise = innovation.rolling(20, min_periods=20).std(ddof=0).shift(1)
        strength = innovation / noise.replace(0.0, np.nan)
        gap = data['open'] / previous - 1.0
        gap_noise = gap.rolling(20, min_periods=20).std(ddof=0).shift(1)
        entry = (
            (gap < -params.gap_sigma * gap_noise)
            & (gap_noise > 0)
            & (strength > 0.25)
            & (close > data['open'])
            & (close < previous)
            & (data['open'] > 0)
            & np.isfinite(gap)
            & np.isfinite(strength)
        ).fillna(False)
        return pd.DataFrame({
            'fractional_close': fractional,
            'fractional_innovation': innovation,
            'innovation_noise': noise,
            'innovation_strength': strength,
            'gap': gap,
            'gap_noise': gap_noise,
            'entry': entry,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                # Evaluate the target at the close; the shifted exit fills next bar.
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or not np.isfinite(closes[i])
                    or closes[i] >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar + 1 >= 2
                ):
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapMemoryParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 102.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    entries = pd.DataFrame({'entry': [True, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], 'close'] = 101.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
