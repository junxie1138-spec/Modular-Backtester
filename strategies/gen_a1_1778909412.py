from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    dd_lookback: int = 40
    dd_threshold: float = 0.03
    range_window: int = 20
    decay: float = 0.55
    capacity: float = 1.5
    fill_frac: float = 0.80
    dd_scale: float = 0.10
    base_size: float = 0.5
    size_gain: float = 0.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778909412"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.dd_lookback, params.range_window)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # --- intrabar pressure from high-low range dynamics ---
        rng = high - low
        safe_rng = rng.where(rng > 0.0, np.nan)
        close_loc = ((close - low) / safe_rng).clip(0.0, 1.0).fillna(0.5)
        pressure = (close_loc - 0.5) * 2.0  # -1 .. +1, who held the bar

        rw = int(max(2, params.range_window))
        avg_rng = rng.rolling(rw, min_periods=rw).mean()
        rel_range = (rng / avg_rng.replace(0.0, np.nan)).clip(0.0, 4.0).fillna(0.0)

        # range-weighted directional deposit into the queue
        deposit = (pressure * rel_range).fillna(0.0).to_numpy(dtype=float)

        # --- leaky integrator with hard capacity cap (queue overflow) ---
        cap = float(abs(params.capacity)) if params.capacity != 0 else 1.0
        decay = float(min(max(params.decay, 0.0), 0.999))
        n = len(deposit)
        buf = np.zeros(n, dtype=float)
        b = 0.0
        for i in range(n):
            b = decay * b + deposit[i]
            if b > cap:
                b = cap
            elif b < -cap:
                b = -cap
            buf[i] = b
        buffer = pd.Series(buf, index=data.index)

        # --- drawdown gate ---
        dl = int(max(2, params.dd_lookback))
        roll_max = close.rolling(dl, min_periods=dl).max()
        dd = (close / roll_max - 1.0)
        in_dd = (dd <= -abs(params.dd_threshold)).fillna(False)

        # --- overflow triggers, two-bar confirmation ---
        trigger = cap * float(min(max(params.fill_frac, 0.05), 0.99))
        long_raw = in_dd & (buffer >= trigger)
        short_raw = in_dd & (buffer <= -trigger)

        long_conf = long_raw & long_raw.shift(1).fillna(False).astype(bool)
        short_conf = short_raw & short_raw.shift(1).fillna(False).astype(bool)

        out = pd.DataFrame(index=data.index)
        out["buffer"] = buffer
        out["dd"] = dd.fillna(0.0)
        out["long_conf"] = long_conf.astype(float)
        out["short_conf"] = short_conf.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        long_conf = indicators["long_conf"].to_numpy(dtype=float) > 0.5
        short_conf = indicators["short_conf"].to_numpy(dtype=float) > 0.5
        dd = indicators["dd"].to_numpy(dtype=float)
        n = len(data)

        # signal-reversal exit: once in, position only changes when the
        # OPPOSITE two-bar-confirmed entry condition fires (flip-only).
        sig = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            if long_conf[i] and pos != 1:
                pos = 1
            elif short_conf[i] and pos != -1:
                pos = -1
            sig[i] = pos

        # size scales with drawdown depth at decision time
        scale = max(float(params.dd_scale), 1e-9)
        depth = np.minimum(np.abs(dd) / scale, 1.0)
        size = float(params.base_size) + float(params.size_gain) * depth
        size = np.clip(size, 0.05, 2.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
