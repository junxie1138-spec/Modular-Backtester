from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidencyParams:
    streak_bars: int = 3
    profit_target: float = 0.012

    def __post_init__(self):
        if self.streak_bars < 1:
            raise ValueError('streak_bars must be positive')
        if not 0.0 < self.profit_target < 1.0:
            raise ValueError('profit_target must be between zero and one')


class GeneratedStrategy(BaseStrategy[ResidencyParams]):
    strategy_id = 'gen_local_1791254417'

    @classmethod
    def params_type(cls):
        return ResidencyParams

    def warmup_bars(self, params: ResidencyParams) -> int:
        return int(21 + params.streak_bars)

    def indicators(self, data: pd.DataFrame, params: ResidencyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ceiling'] = data['high'].rolling(20, min_periods=20).max().shift(1)
        out['floor'] = data['low'].rolling(20, min_periods=20).min().shift(1)
        out['width'] = out['ceiling'] - out['floor']
        out['position'] = (data['close'] - out['floor']) / out['width'].where(out['width'] > 0)
        out['position'] = out['position'].replace([np.inf, -np.inf], np.nan)
        out['resident'] = out['position'].between(0.8, 1.0).fillna(False)
        out['streak'] = out['resident'].astype(int).groupby((~out['resident']).cumsum()).cumsum()
        out['prior_streak'] = out['streak'].shift(1).fillna(0)
        out['entry'] = (
            (out['prior_streak'] >= params.streak_bars)
            & (data['close'] > out['ceiling'])
            & np.isfinite(data['close'])
            & (data['close'] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ResidencyParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if entry_bar < 0 and i > 0 and signal[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                entry_bar = i
                entry_price = opens[i]

            if entry_bar >= 0:
                # Close-confirmed exits execute at the following open.
                expired = i - entry_bar + 1 >= 4
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if expired or target_hit:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1

        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = ResidencyParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [80.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'high': [81.0, 102.0, 102.0, 102.0, 102.0, 102.0],
        'low': [79.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        'close': [80.0, 100.0, 101.3, 100.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    entries = pd.DataFrame({'entry': [True, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    data['close'] = 100.0
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    flat = pd.DataFrame(100.0, index=pd.date_range('2020-01-01', periods=30), columns=['open', 'high', 'low', 'close', 'volume'])
    derived = strategy.indicators(flat, params)
    assert not derived['entry'].any()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()
