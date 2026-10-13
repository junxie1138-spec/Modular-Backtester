from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueParams:
    target_vol: float = 0.12
    atr_multiple: float = 2.5

    def __post_init__(self):
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError('target_vol must be finite and positive')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive')


class GeneratedStrategy(BaseStrategy[QueueParams]):
    strategy_id = 'gen_local_1791317383'

    @classmethod
    def params_type(cls):
        return QueueParams

    def warmup_bars(self, params: QueueParams) -> int:
        return 31

    def indicators(self, data: pd.DataFrame, params: QueueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['daily_vol'] = out['return'].rolling(20, min_periods=20).std(ddof=0)
        out['annual_vol'] = out['daily_vol'] * np.sqrt(252.0)
        out['capacity'] = 0.5 * out['return'].abs().rolling(20, min_periods=20).mean()
        out['workload'] = -out['return'] - out['capacity']
        out['cumulative_workload'] = out['workload'].fillna(0.0).cumsum()
        out['running_floor'] = out['cumulative_workload'].cummin().clip(upper=0.0)
        out['queue'] = (out['cumulative_workload'] - out['running_floor']).clip(lower=0.0).where(out['workload'].notna())
        out['prior_queue'] = out['queue'].shift(1)
        out['prior_peak_queue'] = out['queue'].rolling(10, min_periods=10).max().shift(1)
        out['monthly_return'] = out['return'].rolling(20, min_periods=20).sum()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['entry'] = (
            out['queue'].eq(0.0)
            & out['prior_queue'].gt(0.0)
            & out['prior_peak_queue'].ge(2.0 * out['daily_vol'])
            & out['monthly_return'].gt(0.0)
            & out['daily_vol'].gt(0.0)
            & out['atr'].gt(0.0)
        ).fillna(False)
        out['target_size'] = (
            params.target_vol / out['annual_vol'].where(out['annual_vol'].gt(0.0))
        ).replace([np.inf, -np.inf], np.nan).clip(lower=0.05, upper=0.95).fillna(0.05)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: QueueParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)

        for i in range(len(data)):
            price = closes[i]
            atr = atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # Initialize on the first holding bar, after the entry decision.
                    high_water = price if np.isnan(high_water) else max(high_water, price)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                if not valid or price <= stop or age >= 18:
                    active = False
                    age = 0
                    high_water = np.nan
                    stop = -np.inf
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['target_size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = QueueParams()
    index = pd.date_range('2020-01-01', periods=24, freq='D')
    data = pd.DataFrame({'close': np.full(24, 100.0)}, index=index)
    ind = pd.DataFrame({'atr': np.ones(24), 'entry': np.zeros(24, dtype=bool),
                        'target_size': np.full(24, 0.4)}, index=index)
    ind.loc[index[0], 'entry'] = True
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:19].eq(1).all()
    assert result['signal'].iloc[19:].eq(0).all()
    data.loc[index[1:4], 'close'] = [105.0, 104.0, 100.0]
    ind.loc[index[3], 'atr'] = 10.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    assert result['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
