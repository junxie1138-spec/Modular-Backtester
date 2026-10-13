from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AutocorrelationParams:
    ma_window: int = 12
    corr_window: int = 32
    horizon: int = 4
    entry_z: float = 0.5

    def __post_init__(self):
        if self.ma_window < 2 or self.corr_window < 3 or self.horizon < 1:
            raise ValueError('Invalid lookback lengths')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')


class GeneratedStrategy(BaseStrategy[AutocorrelationParams]):
    strategy_id = 'gen_local_1790968497'

    @classmethod
    def params_type(cls):
        return AutocorrelationParams

    def warmup_bars(self, params: AutocorrelationParams) -> int:
        return int(max(params.ma_window, params.corr_window + 2 * params.horizon))

    def indicators(self, data: pd.DataFrame, params: AutocorrelationParams) -> pd.DataFrame:
        close = data['close'].astype(float).where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].where(out['price_std'] > 0)
        out['block_return'] = close.pct_change(periods=params.horizon, fill_method=None)
        out['lagged_block_return'] = out['block_return'].shift(params.horizon)
        out['autocorrelation'] = out['block_return'].rolling(
            params.corr_window, min_periods=params.corr_window
        ).corr(out['lagged_block_return']).clip(-1.0, 1.0)
        out['forecast'] = out['autocorrelation'] * out['block_return']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AutocorrelationParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators['z']) & np.isfinite(indicators['forecast'])
        # Hold only while both entry primitives agree; their disagreement exits.
        agreement = valid & (indicators['z'] < -params.entry_z) & (indicators['forecast'] > 0.0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = agreement.fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AutocorrelationParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'z': [-1.0, -0.8, -0.2, -1.0, -1.0, np.nan, -1.0],
        'forecast': [0.01, 0.02, 0.01, -0.01, 0.01, 0.01, 0.01],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1, 0]
    assert result.index.equals(index) and (result['size'] > 0).all()
    assert GeneratedStrategy.warmup_bars(None, params) == 40


if __name__ == '__main__':
    _self_check()
