from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePressureParams:
    baseline_window: int = 20
    signature_window: int = 5
    compression_ratio: float = 0.8
    area_threshold: float = 0.04
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangePressureParams]):
    strategy_id = 'gen_local-w2_1791250519'

    @classmethod
    def params_type(cls) -> type[RangePressureParams]:
        return RangePressureParams

    def warmup_bars(self, params: RangePressureParams) -> int:
        if params.baseline_window < 2 or params.signature_window < 3:
            raise ValueError('Indicator windows are too short')
        return int(params.baseline_window + params.signature_window)

    def indicators(self, data: pd.DataFrame, params: RangePressureParams) -> pd.DataFrame:
        if params.baseline_window < 2 or params.signature_window < 3:
            raise ValueError('Indicator windows are too short')
        out = pd.DataFrame(index=data.index)
        width = (data['high'] - data['low']).where(lambda s: s > 0)
        out['relative_range'] = width / data['close'].where(data['close'] > 0)
        out['range_baseline'] = out['relative_range'].rolling(params.baseline_window).mean()
        out['range_ratio'] = out['relative_range'] / out['range_baseline'].where(out['range_baseline'] > 0)
        out['compression'] = -out['range_ratio']
        out['pressure'] = ((2.0 * data['close'] - data['high'] - data['low']) / width).clip(-1.0, 1.0)
        x, y = out['compression'], out['pressure']
        steps = params.signature_window - 1
        out['cross_step'] = x.shift(1) * y - y.shift(1) * x
        # Antisymmetric level-two signature, translated to the window's first point.
        out['signature_area'] = 0.5 * (
            out['cross_step'].rolling(steps).sum()
            - x.shift(steps) * y + y.shift(steps) * x
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out['setup'] = (
            (out['range_ratio'] <= params.compression_ratio)
            & (out['signature_area'] >= params.area_threshold)
            & (out['pressure'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangePressureParams) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError('Require a positive target and a one- or two-bar time stop')
        setup = indicators['setup'].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                held = i - entry_bar + 1
                # Evaluate at the close; the shifted exit executes next bar.
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if invalid_entry or target_hit or held >= params.max_hold_bars:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    bars = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'setup': [False, True, True, False, False, False, False]}, index=index)
    params = RangePressureParams()
    result = GeneratedStrategy.generate_signals(None, bars, features, None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0]
    bars.loc[index[3], 'close'] = 101.0
    result = GeneratedStrategy.generate_signals(None, bars, features, None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 0, 0, 0]
