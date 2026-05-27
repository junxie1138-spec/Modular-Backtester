from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolBudgetZScoreParams:
    ma_window: int = 40
    z_window: int = 20
    z_lower: float = 0.5
    z_upper: float = 2.0
    vol_window: int = 20
    target_vol_annual: float = 0.15
    budget_capacity: float = 0.06
    profit_target: float = 0.03
    time_stop_bars: int = 10
    size_floor: float = 0.4
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[VolBudgetZScoreParams]):
    strategy_id = "gen_a1_1779648853"

    @classmethod
    def params_type(cls):
        return VolBudgetZScoreParams

    def warmup_bars(self, params: VolBudgetZScoreParams) -> int:
        return int(max(params.ma_window, params.z_window, params.vol_window)) + 5

    def indicators(self, data: pd.DataFrame, params: VolBudgetZScoreParams) -> pd.DataFrame:
        close = data["close"].astype(float)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        dist = close - ma
        dist_mean = dist.rolling(params.z_window, min_periods=params.z_window).mean()
        dist_std = dist.rolling(params.z_window, min_periods=params.z_window).std(ddof=0)
        z = (dist - dist_mean) / dist_std.replace(0.0, np.nan)

        log_close = np.log(close.where(close > 0.0))
        log_ret = log_close.diff()
        realized_vol = (
            log_ret.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
            * np.sqrt(252.0)
        )

        target = float(params.target_vol_annual)
        headroom = (target - realized_vol).clip(lower=-target, upper=target)

        out = pd.DataFrame(
            {
                "ma": ma,
                "z": z,
                "realized_vol": realized_vol,
                "headroom": headroom,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolBudgetZScoreParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        rv = indicators["realized_vol"].to_numpy(dtype=float)
        head = indicators["headroom"].to_numpy(dtype=float)

        raw_signal = np.zeros(n, dtype=np.int64)
        size_arr = np.zeros(n, dtype=float)

        capacity = float(params.budget_capacity)
        z_lo = float(params.z_lower)
        z_hi = float(params.z_upper)
        pt = float(params.profit_target)
        ts = int(params.time_stop_bars)
        size_floor = float(params.size_floor)
        size_cap = float(params.size_cap)
        target_vol = float(params.target_vol_annual)

        budget = 0.0
        in_pos = False
        entry_price = 0.0
        bars_held = 0
        active_size = 0.0

        for i in range(n):
            h = head[i]
            if np.isfinite(h):
                budget = budget + h
                if budget < 0.0:
                    budget = 0.0
            overflow = budget >= capacity
            if overflow:
                budget = 0.0

            if in_pos:
                bars_held += 1
                ret_since_entry = (close[i] / entry_price) - 1.0 if entry_price > 0.0 else 0.0
                exit_now = (ret_since_entry >= pt) or (bars_held >= ts)
                if exit_now:
                    raw_signal[i] = 0
                    size_arr[i] = 0.0
                    in_pos = False
                    bars_held = 0
                    entry_price = 0.0
                    active_size = 0.0
                else:
                    raw_signal[i] = 1
                    size_arr[i] = active_size
            else:
                zi = z[i]
                rvi = rv[i]
                z_ok = np.isfinite(zi) and (z_lo <= zi <= z_hi)
                vol_ok = np.isfinite(rvi) and rvi > 1e-6
                if overflow and z_ok and vol_ok:
                    sz = target_vol / rvi
                    if not np.isfinite(sz):
                        sz = size_floor
                    if sz < size_floor:
                        sz = size_floor
                    elif sz > size_cap:
                        sz = size_cap
                    raw_signal[i] = 1
                    size_arr[i] = sz
                    in_pos = True
                    entry_price = close[i]
                    active_size = sz
                    bars_held = 0
                else:
                    raw_signal[i] = 0
                    size_arr[i] = 0.0

        df = pd.DataFrame(
            {
                "signal": raw_signal,
                "size": size_arr,
            },
            index=idx,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
