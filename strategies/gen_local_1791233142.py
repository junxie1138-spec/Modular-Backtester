from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    shock_threshold: float = 4.5
    hold_bars: int = 8

    def __post_init__(self):
        if not np.isfinite(self.shock_threshold) or self.shock_threshold <= 0:
            raise ValueError('shock_threshold must be finite and positive')
        if not isinstance(self.hold_bars, int) or isinstance(self.hold_bars, bool) or self.hold_bars < 1:
            raise ValueError('hold_bars must be a positive integer')


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = 'gen_local_1791233142'

    @classmethod
    def params_type(cls):
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['return'] = close.pct_change(fill_method=None)
        out['prior_volatility'] = out['return'].rolling(20, min_periods=20).std(ddof=0).shift(1)
        out['prior_volume'] = volume.rolling(20, min_periods=20).mean().shift(1)
        out['relative_volume'] = volume / out['prior_volume'].where(out['prior_volume'] > 0)
        out['standardized_return'] = out['return'] / out['prior_volatility'].where(out['prior_volatility'] > 0)
        out['pressure'] = out['standardized_return'].clip(-4.0, 4.0) * out['relative_volume'].clip(0.0, 3.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        upward = np.zeros(len(data), dtype=float)
        downward = np.zeros(len(data), dtype=float)
        event = np.zeros(len(data), dtype=int)
        positive = negative = 0.0
        pressure = out['pressure'].to_numpy()
        participation = out['relative_volume'].to_numpy()
        returns = out['return'].to_numpy()
        for i, value in enumerate(pressure):
            if not np.isfinite(value):
                positive = negative = 0.0
                continue
            positive = max(0.0, positive + value - 0.5)
            negative = min(0.0, negative + value + 0.5)
            upward[i] = positive
            downward[i] = negative
            if positive >= params.shock_threshold or negative <= -params.shock_threshold:
                if participation[i] >= 1.2:
                    if positive >= params.shock_threshold and returns[i] > 0:
                        event[i] = 1
                    elif negative <= -params.shock_threshold and returns[i] < 0:
                        event[i] = -1
                positive = negative = 0.0
        out['upward_cusum'] = upward
        out['downward_cusum'] = downward
        out['event'] = event
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, event in enumerate(indicators['event'].to_numpy(dtype=int)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signal[i] = position
                continue
            if event != 0:
                position = int(event)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame(index=index)
    events = pd.DataFrame({'event': [1, -1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, events, None, ShockParams(hold_bars=5)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1]
    assert result['size'].gt(0).all()
