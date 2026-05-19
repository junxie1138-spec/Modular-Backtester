from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 200
    vol_window: int = 20
    vol_baseline_window: int = 120
    vol_z_entry: float = -0.10
    profit_target: float = 0.02
    max_hold: int = 4
    size_scale: float = 0.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779146879"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        seasonal = int(params.vol_window) + int(params.vol_baseline_window)
        return int(max(int(params.ma_window), seasonal)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        idx = data.index
        close = data["close"]

        out = pd.DataFrame(index=idx)

        # --- regime filter: 200-day moving average (the mandatory twist) ---
        ma = close.rolling(int(params.ma_window), min_periods=int(params.ma_window)).mean()
        out["ma"] = ma
        out["regime_ok"] = (close > ma).astype(float)

        # --- volatility primitive: realized-vol z-score vs its own baseline ---
        ret = close.pct_change()
        rv = ret.rolling(int(params.vol_window), min_periods=int(params.vol_window)).std()
        base_mean = rv.rolling(
            int(params.vol_baseline_window), min_periods=int(params.vol_baseline_window)
        ).mean()
        base_std = rv.rolling(
            int(params.vol_baseline_window), min_periods=int(params.vol_baseline_window)
        ).std()
        base_std = base_std.replace(0.0, np.nan)
        vol_z = (rv - base_mean) / base_std
        vol_z = vol_z.replace([np.inf, -np.inf], np.nan)
        out["vol_z"] = vol_z

        # --- seasonality: detect the monthly options-expiration week ---
        # OpEx week = the Mon-Fri week containing the third Friday of the month.
        first_of_month = idx.to_period("M").to_timestamp()
        first_weekday = np.asarray(first_of_month.weekday)  # 0=Mon .. 6=Sun
        first_friday_dom = 1 + ((4 - first_weekday) % 7)
        third_friday_dom = first_friday_dom + 14
        monday_dom = third_friday_dom - 4
        day = np.asarray(idx.day)
        in_opex_week = (day >= monday_dom) & (day <= third_friday_dom)

        prev_in = np.concatenate(([False], in_opex_week[:-1]))
        entry_trigger = in_opex_week & (~prev_in)
        out["in_opex_week"] = in_opex_week.astype(float)
        out["entry_trigger"] = entry_trigger.astype(float)

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)

        trigger = indicators["entry_trigger"].to_numpy() > 0.5
        regime_ok = indicators["regime_ok"].to_numpy() > 0.5
        vol_z = indicators["vol_z"].to_numpy(dtype=float)

        z_thresh = float(params.vol_z_entry)
        pt = float(params.profit_target)
        max_hold = max(1, int(params.max_hold))
        size_scale = float(params.size_scale)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        entry_size = 1.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= max_hold:
                    signal[i] = 0
                    in_pos = False
                    bars_held = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                z = vol_z[i]
                vol_compressed = np.isfinite(z) and z < z_thresh
                if trigger[i] and regime_ok[i] and vol_compressed:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    depth = max(0.0, -z)
                    entry_size = float(np.clip(1.0 + depth * size_scale, 0.5, 2.0))
                    signal[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        size_series = pd.Series(size, index=data.index)
        size_series = size_series.where(size_series > 0.0, 1.0)
        df["size"] = size_series

        return SignalFrame(data=df, signal_column="signal", size_column="size")
