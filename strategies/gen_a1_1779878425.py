from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_period: int = 200
    vol_ma_period: int = 20
    gap_threshold_pct: float = 0.0015
    vol_ratio_threshold: float = 1.20
    leak_rate: float = 0.15
    capacity_threshold: float = 0.006
    profit_target_pct: float = 0.030
    time_stop_bars: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779878425"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # 200-day regime MA dominates; pad for vol baseline and the single NaN
        # introduced by the prev-close shift used in gap computation.
        return int(params.ma_period) + int(params.vol_ma_period) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        # Overnight gap as fraction of prior close.
        prev_close = close.shift(1)
        gap_pct = (open_ / prev_close) - 1.0

        # Rolling volume baseline; ratio measures relative participation.
        vol_ma = volume.rolling(int(params.vol_ma_period), min_periods=int(params.vol_ma_period)).mean()
        vol_ratio = volume / vol_ma.replace(0.0, np.nan)

        # Token-bucket arrival: positive only on a volume-confirmed gap-up.
        gap_ok = gap_pct > float(params.gap_threshold_pct)
        vol_ok = vol_ratio > float(params.vol_ratio_threshold)
        token_raw = (gap_pct * vol_ratio).where(gap_ok & vol_ok, 0.0)
        token = token_raw.fillna(0.0).to_numpy(dtype=float)

        # Leaky-bucket reservoir: continuous exponential decay, no hard cap.
        # R[t] = max(0, R[t-1] * (1 - leak) + token[t]).  Path-dependent -> loop.
        leak = float(params.leak_rate)
        if leak < 0.0:
            leak = 0.0
        elif leak > 1.0:
            leak = 1.0
        n = len(token)
        reservoir = np.zeros(n, dtype=float)
        prev = 0.0
        for i in range(n):
            cur = prev * (1.0 - leak) + token[i]
            if cur < 0.0:
                cur = 0.0
            reservoir[i] = cur
            prev = cur

        # 200-day regime filter (mandated twist).
        ma200 = close.rolling(int(params.ma_period), min_periods=int(params.ma_period)).mean()

        out = pd.DataFrame(index=data.index)
        out["gap_pct"] = gap_pct.fillna(0.0)
        out["vol_ratio"] = vol_ratio.fillna(0.0)
        out["token"] = token
        out["reservoir"] = reservoir
        out["ma200"] = ma200
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        reservoir = indicators["reservoir"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)

        cap = float(params.capacity_threshold)
        pt = float(params.profit_target_pct)
        time_stop = int(params.time_stop_bars)

        n = len(close)
        sig = np.zeros(n, dtype=int)

        in_pos = False
        entry_px = 0.0
        bars_held = 0

        for i in range(n):
            regime_ok = (not np.isnan(ma200[i])) and close[i] > ma200[i]
            saturated = reservoir[i] >= cap

            if in_pos:
                bars_held += 1
                px = close[i]
                ret = (px / entry_px) - 1.0 if entry_px > 0.0 else 0.0
                # Profit-target OR time-stop -> exit (whichever fires first).
                if ret >= pt or bars_held >= time_stop:
                    in_pos = False
                    bars_held = 0
                    entry_px = 0.0
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                if saturated and regime_ok and not np.isnan(close[i]):
                    in_pos = True
                    entry_px = close[i]
                    bars_held = 0
                    sig[i] = 1
                else:
                    sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
