from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_days: int = 3
    acceleration_floor: float = 0.001
    month_sessions: int = 3
    hold_bars: int = 2

    def __post_init__(self):
        if self.roc_days < 1:
            raise ValueError('roc_days must be positive')
        if not np.isfinite(self.acceleration_floor) or self.acceleration_floor < 0:
            raise ValueError('acceleration_floor must be finite and nonnegative')
        if not 1 <= self.month_sessions <= 5:
            raise ValueError('month_sessions must be between 1 and 5')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791292887'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.roc_days + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['roc'] = close.pct_change(params.roc_days, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['acceleration'] = out['roc'].diff().replace([np.inf, -np.inf], np.nan)
        out['month_key'] = data.index.year * 12 + data.index.month
        out['month_session'] = out.groupby('month_key', sort=False).cumcount() + 1
        # Suppress the first observed month: its initial sessions may be missing.
        out['complete_month_start'] = out['month_key'].ne(out['month_key'].iloc[0]) if len(out) else False
        out['entry_ready'] = (
            out['complete_month_start']
            & out['month_session'].le(params.month_sessions)
            & close.gt(out['ma200'])
            & out['roc'].lt(0)
            & out['acceleration'].gt(params.acceleration_floor)
            & np.isfinite(close)
            & np.isfinite(out['ma200'])
            & np.isfinite(out['roc'])
            & np.isfinite(out['acceleration'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(ready):
            if remaining:
                signal[i] = 1
                remaining -= 1
            elif i and signal[i - 1] == 1:
                # Mandatory expiry bar; fresh entries cannot postpone the exit.
                continue
            elif enter:
                signal[i] = 1
                remaining = params.hold_bars - 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10, freq='B')
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = pd.DataFrame({'entry_ready': True}, index=index)
    strategy = GeneratedStrategy()
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
    ):
        params = Params(hold_bars=hold)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == expected
        assert result['size'].gt(0).all()
        assert not strategy.indicators(data, params)['entry_ready'].any()
        assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
