from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VRRegimeParams:
    vr_window: int = 60
    vr_lag: int = 5
    trigger_window: int = 5
    vr_band: float = 0.15
    size_floor: float = 0.4
    size_cap: float = 1.5
    size_scale: float = 4.0
    min_abs_return: float = 0.001


class GeneratedStrategy(BaseStrategy):
    strategy_id = 'gen_a1_1779652364'

    @classmethod
    def params_type(cls):
        return VRRegimeParams

    @classmethod
    def warmup_bars(cls, params: VRRegimeParams) -> int:
        return int(params.vr_window + params.vr_lag + params.trigger_window + 2)

    def indicators(self, data: pd.DataFrame, params: VRRegimeParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        r1 = close.pct_change()
        rk = close.pct_change(params.vr_lag)

        var1 = r1.rolling(params.vr_window, min_periods=params.vr_window).var()
        vark = rk.rolling(params.vr_window, min_periods=params.vr_window).var()

        denom = float(params.vr_lag) * var1
        denom = denom.replace(0.0, np.nan)
        vr = vark / denom

        trig = close.pct_change(params.trigger_window)

        out = pd.DataFrame(index=data.index)
        out['vr'] = vr
        out['trig'] = trig
        return out

    def generate_signals(self, data, indicators, ctx, params):
        vr = indicators['vr']
        trig = indicators['trig']

        plastic = (vr >= (1.0 + params.vr_band)).fillna(False)
        elastic = (vr <= (1.0 - params.vr_band)).fillna(False)

        trig_sign = pd.Series(np.sign(trig.fillna(0.0).to_numpy()), index=data.index)

        entry = pd.Series(0.0, index=data.index)
        entry = entry.where(~plastic, trig_sign)
        entry = entry.where(~elastic, -trig_sign)

        weak = (trig.abs() < params.min_abs_return).fillna(True)
        entry = entry.where(~weak, 0.0)
        entry = entry.fillna(0.0).astype(int)

        vr_dev = (vr - 1.0).abs().fillna(0.0)
        raw_size = float(params.size_floor) + float(params.size_scale) * vr_dev
        raw_size = raw_size.clip(lower=float(params.size_floor), upper=float(params.size_cap))
        raw_size = raw_size.fillna(float(params.size_floor))

        entry_arr = entry.to_numpy()
        raw_size_arr = raw_size.to_numpy()
        n = len(data)

        pos = np.zeros(n, dtype=np.int64)
        size_arr = np.full(n, float(params.size_floor), dtype=np.float64)

        for i in range(n):
            prev_pos = int(pos[i - 1]) if i > 0 else 0
            prev_size = size_arr[i - 1] if i > 0 else float(params.size_floor)
            e = int(entry_arr[i])

            if prev_pos == 0:
                if e != 0:
                    pos[i] = e
                    size_arr[i] = float(raw_size_arr[i])
                else:
                    pos[i] = 0
                    size_arr[i] = float(params.size_floor)
            elif e == 0 or e == prev_pos:
                pos[i] = prev_pos
                size_arr[i] = prev_size
            else:
                pos[i] = e
                size_arr[i] = float(raw_size_arr[i])

        df = data.copy()
        df['signal'] = pos
        df['size'] = size_arr

        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(float(params.size_floor))

        return SignalFrame(data=df, signal_column='signal', size_column='size')
