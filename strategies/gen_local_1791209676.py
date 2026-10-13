from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnchorParams:
    lookback: int = 20
    entry_z: float = 1.25

    def __post_init__(self):
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError('lookback must be an integer >= 2')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')


class GeneratedStrategy(BaseStrategy[AnchorParams]):
    strategy_id = 'gen_local_1791209676'

    @classmethod
    def params_type(cls) -> type[AnchorParams]:
        return AnchorParams

    def warmup_bars(self, params: AnchorParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: AnchorParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        window = close.rolling(params.lookback, min_periods=params.lookback)
        out['ma'] = window.mean()
        out['std'] = window.std(ddof=0).replace(0.0, np.nan)
        out['distance'] = close - out['ma']
        out['z'] = (out['distance'] / out['std']).replace([np.inf, -np.inf], np.nan)
        out['previous_z'] = out['z'].shift(1)
        out['previous_ma'] = out['ma'].shift(1)
        out['price_change'] = close.diff()
        out['distance_change'] = out['distance'].diff()
        out['expired_close'] = close.shift(params.lookback)
        out['entry'] = (
            (out['previous_z'] <= -params.entry_z)
            & (out['z'] > out['previous_z'])
            & (out['z'] < 0.0)
            & (out['price_change'] <= 0.0)
            & (out['distance_change'] > 0.0)
            & (out['expired_close'] > out['previous_ma'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnchorParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_at = -1
        for i, entry in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif entry:
                signal[i] = 1
                exit_at = i + 2
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, AnchorParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert (result.data['size'] > 0.0).all()
