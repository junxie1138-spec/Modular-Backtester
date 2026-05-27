from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapQueueParams:
    gap_pct_threshold: float = 0.003
    volume_lookback: int = 20
    volume_z_threshold: float = 1.0
    gap_inventory_window: int = 30
    gap_inventory_capacity: int = 4
    profit_target_pct: float = 0.03
    time_stop_bars: int = 5


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779649243"

    @classmethod
    def params_type(cls):
        return GapQueueParams

    @classmethod
    def warmup_bars(cls, params):
        return int(max(params.volume_lookback, params.gap_inventory_window)) + 2

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        gap = (data["open"] - prev_close) / prev_close
        out["gap"] = gap

        vol = data["volume"].astype(float)
        vol_mean = vol.shift(1).rolling(
            params.volume_lookback, min_periods=params.volume_lookback
        ).mean()
        vol_std = vol.shift(1).rolling(
            params.volume_lookback, min_periods=params.volume_lookback
        ).std()
        vol_z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)
        out["vol_z"] = vol_z

        gap_up = (gap > params.gap_pct_threshold).fillna(False)
        vol_ok = (vol_z > params.volume_z_threshold).fillna(False)
        out["gap_up"] = gap_up.astype(float)
        out["vol_ok"] = vol_ok.astype(float)

        n = len(data)
        unfilled_count = np.zeros(n, dtype=float)
        gap_up_arr = gap_up.to_numpy(dtype=bool)
        prev_close_arr = prev_close.to_numpy(dtype=float)
        low_arr = data["low"].to_numpy(dtype=float)
        window = int(params.gap_inventory_window)
        for i in range(n):
            start = i - window
            if start < 0:
                start = 0
            cnt = 0
            for j in range(start, i):
                if not gap_up_arr[j]:
                    continue
                pc = prev_close_arr[j]
                if np.isnan(pc):
                    continue
                slc = low_arr[j:i]
                if slc.size == 0:
                    continue
                slc_min = np.nanmin(slc)
                if not np.isnan(slc_min) and slc_min > pc:
                    cnt += 1
            unfilled_count[i] = cnt
        out["unfilled_gaps"] = unfilled_count
        out["capacity_ok"] = (unfilled_count < params.gap_inventory_capacity).astype(float)

        primitive_a = gap_up.to_numpy(dtype=bool) & vol_ok.to_numpy(dtype=bool)
        primitive_b = unfilled_count < params.gap_inventory_capacity
        out["entry_raw"] = (primitive_a & primitive_b).astype(float)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].fillna(0.0).to_numpy(dtype=float)

        intent = np.zeros(n, dtype=int)
        in_pos = False
        entry_close = 0.0
        bars_in_pos = 0

        target = float(params.profit_target_pct)
        max_bars = int(params.time_stop_bars)

        for i in range(n):
            if in_pos:
                bars_in_pos += 1
                ret = (close[i] / entry_close) - 1.0 if entry_close > 0 else 0.0
                if ret >= target or bars_in_pos >= max_bars:
                    intent[i] = 0
                    in_pos = False
                    entry_close = 0.0
                    bars_in_pos = 0
                else:
                    intent[i] = 1
            else:
                if entry_raw[i] > 0.5 and not np.isnan(close[i]):
                    in_pos = True
                    entry_close = close[i]
                    bars_in_pos = 0
                    intent[i] = 1
                else:
                    intent[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(intent, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
