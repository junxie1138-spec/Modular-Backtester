from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 12
    profit_target: float = 0.008

    def __post_init__(self):
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 6:
            raise ValueError('lookback must be an integer >= 6')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1790905673'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        out['log_close'] = np.log(close)
        price = out['log_close']
        slope = (price.shift(1) - price).to_numpy(dtype=float)
        upper_barrier = slope.copy()
        lower_barrier = slope.copy()
        upper_degree = np.ones(len(data), dtype=float)
        lower_degree = np.ones(len(data), dtype=float)
        # Strict natural visibility: intermediate closes lie below or above the edge.
        for distance in range(2, params.lookback):
            slope = ((price.shift(distance) - price) / distance).to_numpy(dtype=float)
            upper_degree += slope > upper_barrier
            lower_degree += slope < lower_barrier
            upper_barrier = np.maximum(upper_barrier, slope)
            lower_barrier = np.minimum(lower_barrier, slope)
        ready = price.rolling(params.lookback, min_periods=params.lookback).count().eq(params.lookback)
        out['upper_degree'] = np.where(ready, upper_degree, np.nan)
        out['lower_degree'] = np.where(ready, lower_degree, np.nan)
        out['range_low'] = close.rolling(params.lookback, min_periods=params.lookback).min()
        out['range_high'] = close.rolling(params.lookback, min_periods=params.lookback).max()
        width = (out['range_high'] - out['range_low']).replace(0.0, np.nan)
        out['relative_position'] = (close - out['range_low']) / width
        out['close_change'] = close.diff()
        out['previous_lower_degree'] = out['lower_degree'].shift(1)
        out['entry'] = (
            out['previous_lower_degree'].ge(4)
            & out['lower_degree'].le(2)
            & out['upper_degree'].ge(3)
            & out['relative_position'].le(0.5)
            & out['close_change'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                # A decision on the preceding close enters at this bar's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if not np.isfinite(entry_price) or entry_price <= 0 or target_hit or held >= 2:
                    active = False
                    held = 0
                    continue
                raw[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                held = 0
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=24)
    data = pd.DataFrame({name: 100.0 for name in ('open', 'high', 'low', 'close', 'volume')}, index=index)
    data.loc[index[5], 'close'] = 101.0
    flags = pd.DataFrame({'entry': False}, index=index)
    flags.loc[index[[0, 4]], 'entry'] = True
    result = strategy.generate_signals(data, flags, None, params).data
    # First trade times out; second reaches its target on its first session.
    assert result['signal'].iloc[:8].tolist() == [0, 1, 1, 0, 0, 1, 0, 0]
    assert result['signal'].isin([0, 1]).all() and result['size'].gt(0).all()
    data['close'] = np.exp(np.array([1., 4., 2., 3., 0., 2.] * 4))
    actual = strategy.indicators(data, params)
    window = np.log(data['close'].iloc[-params.lookback:].to_numpy())
    for name, upper in (('upper_degree', True), ('lower_degree', False)):
        expected = 0
        for distance in range(1, len(window)):
            visible = True
            for middle in range(1, distance):
                line = window[-1] + (window[-1-distance] - window[-1]) * middle / distance
                visible = visible and (window[-1-middle] < line if upper else window[-1-middle] > line)
            expected += int(visible)
        assert actual[name].iloc[-1] == expected
    assert actual.iloc[:18].equals(strategy.indicators(data.iloc[:18], params))
