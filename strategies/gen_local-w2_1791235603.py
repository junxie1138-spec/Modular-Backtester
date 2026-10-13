from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAbsorptionParams:
    return_window: int = 6
    atr_multiple: float = 2.0


class GeneratedStrategy(BaseStrategy[GapAbsorptionParams]):
    strategy_id = "gen_local-w2_1791235603"

    @classmethod
    def params_type(cls) -> type[GapAbsorptionParams]:
        return GapAbsorptionParams

    def warmup_bars(self, params: GapAbsorptionParams) -> int:
        return max(max(2, int(params.return_window)) + 1, 15)

    def indicators(self, data: pd.DataFrame, params: GapAbsorptionParams) -> pd.DataFrame:
        window = max(2, int(params.return_window))
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda x: x > 0)
        out['return'] = (data['close'] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['previous_return'] = out['return'].shift(1)
        out['gap'] = (data['open'] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['true_range'] = pd.concat([
            (data['high'] - data['low']).abs(),
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['net_return'] = out['return'].rolling(window, min_periods=window).sum()
        out['loss_load'] = (-out['return'].clip(upper=0)).rolling(window, min_periods=window).sum()
        absorbed = ((out['gap'] < 0) & (out['return'] > 0)).astype(float)
        absorbed = absorbed.where(out['gap'].notna() & out['return'].notna())
        out['absorbed_gap_count'] = absorbed.rolling(window, min_periods=window).sum()
        ready = out[['return', 'previous_return', 'atr', 'net_return', 'loss_load', 'absorbed_gap_count']].notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['atr'] > 0)
            & (out['absorbed_gap_count'] >= (window + 1) // 2)
            & (out['net_return'] > out['loss_load'])
            & (out['return'] > 0)
            & (out['previous_return'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapAbsorptionParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        multiple = max(0.1, float(params.atr_multiple))
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                high_water = max(high_water, close[i])
                previous_stop = stop
                stop = max(stop, high_water - multiple * atr[i])
                assert stop >= previous_stop
                if close[i] <= stop or age >= 10:
                    active = False
                    continue
                signals[i] = 1
            elif i >= warmup - 1 and entry[i] and np.isfinite(close[i]) and close[i] > 0:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        assert df['signal'].isin([0, 1]).all() and df['size'].gt(0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
