from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    shock_z: float = 1.8
    trail_atr: float = 2.0

    def __post_init__(self):
        if not np.isfinite(self.shock_z) or self.shock_z <= 0:
            raise ValueError('shock_z must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = 'gen_local_1791263331'

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        returns = close.pct_change(fill_method=None)
        prior_pressure = returns.shift(1).rolling(2, min_periods=2).sum()
        return_scale = np.sqrt(returns.pow(2).rolling(20, min_periods=20).mean()).shift(1)
        shock = returns.div(return_scale.where(return_scale > 0))
        absolute_flow = returns.abs().rolling(3, min_periods=3).sum()
        shock_share = returns.abs().div(absolute_flow.where(absolute_flow > 0))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        valid = (
            shock.abs().ge(params.shock_z)
            & shock_share.ge(0.65)
            & (returns * prior_pressure).lt(0)
            & atr.gt(0)
        )
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[valid & returns.lt(0)] = 1
        entry.loc[valid & returns.gt(0)] = -1
        return pd.DataFrame({
            'returns': returns,
            'prior_pressure': prior_pressure,
            'return_scale': return_scale,
            'absolute_flow': absolute_flow,
            'shock_share': shock_share,
            'shock': shock,
            'true_range': true_range,
            'atr': atr,
            'entry': entry,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        high_water = low_water = high_stop = low_stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if side:
                age += 1
                if not valid:
                    side = 0
                else:
                    # The position fills on this bar when age becomes one.
                    if age == 1:
                        high_water = low_water = close
                        high_stop = close - params.trail_atr * atr
                        low_stop = close + params.trail_atr * atr
                    else:
                        high_water = max(high_water, close)
                        low_water = min(low_water, close)
                        high_stop = max(high_stop, high_water - params.trail_atr * atr)
                        low_stop = min(low_stop, low_water + params.trail_atr * atr)
                    # Apply the required high-water drawdown exit to both sides;
                    # shorts also receive a ratcheting adverse-move stop.
                    if close <= high_stop or (side == -1 and close >= low_stop) or age >= 4:
                        side = 0
                signals[i] = side
                continue
            if valid and entries[i] in (-1, 1):
                side = int(entries[i])
                age = 0
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = ShockParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [100, 101, 103, 101, 100, 100, 100]}, index=index)
    indicators = pd.DataFrame({'atr': [1, 1, 1, 10, 1, 1, 1],
                               'entry': [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data['close'] = 100.0
    indicators['atr'] = 1.0
    indicators['entry'] = [-1, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, -1, 0, 0]
    data['close'] = [100, 100, 97, 97, 97, 97, 97]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0, 0, 0]
