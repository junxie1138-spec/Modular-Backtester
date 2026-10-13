from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePressureParams:
    squeeze_bars: int = 3
    floor_pressure: float = 0.35
    profit_target: float = 0.009
    max_hold_bars: int = 2

    def __post_init__(self):
        if not isinstance(self.squeeze_bars, int) or not 1 <= self.squeeze_bars <= 8:
            raise ValueError('squeeze_bars must be an integer between 1 and 8')
        if not np.isfinite(self.floor_pressure) or not 0 < self.floor_pressure <= 1:
            raise ValueError('floor_pressure must be in (0, 1]')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite')
        if not isinstance(self.max_hold_bars, int) or not 1 <= self.max_hold_bars <= 2:
            raise ValueError('max_hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[RangePressureParams]):
    strategy_id = 'gen_local-w2_1791234415'

    @classmethod
    def params_type(cls):
        return RangePressureParams

    def warmup_bars(self, params: RangePressureParams) -> int:
        # One difference, a rolling window, and one prior-bar offset.
        return int(params.squeeze_bars + 2)

    def indicators(self, data: pd.DataFrame, params: RangePressureParams) -> pd.DataFrame:
        n = params.squeeze_bars
        high = data['high']
        low = data['low']
        close = data['close']
        span = (high - low).where(high > low)
        base = span.shift(n + 1)
        low_step = low.diff()
        rising = low_step.gt(0).where(low_step.notna())
        out = pd.DataFrame(index=data.index)
        out['range'] = span
        out['prior_range'] = span.shift(1)
        out['base_range'] = base
        out['floor_pressure'] = (low.shift(1) - low.shift(n + 1)) / base
        out['ceiling_drift'] = (high.shift(1) - high.shift(n + 1)) / base
        out['rising_floor'] = rising.rolling(n, min_periods=n).min().shift(1)
        out['close_location'] = (close - low) / span
        ready = np.isfinite(out).all(axis=1)
        valid_prices = np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        valid_prices &= data[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
        out['ready'] = ready & valid_prices
        out['entry'] = (
            out['ready']
            & out['rising_floor'].eq(1)
            & out['floor_pressure'].ge(params.floor_pressure)
            & out['ceiling_drift'].le(0.20)
            & out['prior_range'].lt(out['base_range'])
            & low.ge(low.shift(1))
            & high.gt(high.shift(1) + 0.10 * out['prior_range'])
            & out['close_location'].ge(0.75)
            & close.gt(data['open'])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangePressureParams,
    ) -> SignalFrame:
        entries = indicators['entry'].to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Capture the entry open only after that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                # Closing target decisions execute at the following open.
                target_hit = valid and closes[i] >= entry_price * (1 + params.profit_target)
                if not valid or target_hit or age >= params.max_hold_bars:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif entries[i]:
                desired[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({'signal': desired, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = RangePressureParams()
    data = pd.DataFrame({
        'open': [102, 102, 102, 102, 104, 104.5, 104, 105],
        'high': [104, 104, 104, 104, 105, 105.5, 105, 106],
        'low': [100, 100.5, 101, 101.5, 102, 103, 102, 103],
        'close': [102, 102.5, 103, 103.5, 104.8, 105, 104.5, 105],
        'volume': [1000] * 8,
    }, index=pd.date_range('2024-01-02', periods=8, freq='B'))
    features = strategy.indicators(data, params)
    result = strategy.generate_signals(data, features, None, params).data
    assert features['entry'].tolist() == [False, False, False, False, True, False, False, False]
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    target_data = data.copy()
    target_data.loc[data.index[5], 'close'] = 105.49
    target_features = strategy.indicators(target_data, params)
    target_result = strategy.generate_signals(target_data, target_features, None, params).data
    assert target_result['signal'].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    prefix = data.iloc[:6]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert prefix_result.equals(result.iloc[:6])
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert result['signal'].isin([0, 1]).all() and result['size'].gt(0).all()
    assert strategy.warmup_bars(RangePressureParams(squeeze_bars=8)) == 10


if __name__ == '__main__':
    _self_check()
