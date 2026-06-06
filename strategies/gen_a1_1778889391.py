from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePressureParams:
    loc_window: int = 5
    range_short: int = 5
    range_long: int = 20
    entry_threshold: float = 0.25
    vol_window: int = 20
    target_vol: float = 0.15
    max_size: float = 1.5
    min_size: float = 0.4


class GeneratedStrategy(BaseStrategy[RangePressureParams]):
    strategy_id = "gen_a1_1778889391"

    @classmethod
    def params_type(cls) -> type[RangePressureParams]:
        return RangePressureParams

    @staticmethod
    def warmup_bars(params: RangePressureParams) -> int:
        return int(max(params.loc_window, params.range_long, params.vol_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangePressureParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        rng = high - low
        rng_pos = rng.where(rng > 0.0, np.nan)
        mid = (high + low) / 2.0

        # Intrabar close location: -1 at the bar low, +1 at the bar high.
        z = (close - mid) / (rng_pos / 2.0)
        z = z.clip(-1.0, 1.0).fillna(0.0)

        w = max(int(params.loc_window), 1)
        rng_filled = rng.fillna(0.0)
        # Range-weighted average close location: wide-range bars dominate.
        num = (z * rng_filled).rolling(w, min_periods=w).sum()
        den = rng_filled.rolling(w, min_periods=w).sum()
        pressure = num / den.where(den > 0.0, np.nan)

        rs = max(int(params.range_short), 1)
        rl = max(int(params.range_long), rs + 1)
        rmean_s = rng.rolling(rs, min_periods=rs).mean()
        rmean_l = rng.rolling(rl, min_periods=rl).mean()
        range_ratio = rmean_s / rmean_l.where(rmean_l > 0.0, np.nan)

        vw = max(int(params.vol_window), 2)
        ret = close.pct_change()
        rv = ret.rolling(vw, min_periods=vw).std() * np.sqrt(252.0)
        size_vol = float(params.target_vol) / rv.where(rv > 0.0, np.nan)

        out = pd.DataFrame(index=data.index)
        out["pressure"] = pressure
        out["range_ratio"] = range_ratio
        out["size_vol"] = size_vol
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangePressureParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        range_ratio = indicators["range_ratio"].to_numpy(dtype=float)
        size_vol = indicators["size_vol"].to_numpy(dtype=float)

        t = float(params.entry_threshold)
        raw = np.zeros(n, dtype=np.int64)
        state = 0
        for i in range(n):
            p = pressure[i]
            if not np.isfinite(p):
                raw[i] = state
                continue
            if state == 0:
                # Symmetric entry: decisively positive range-weighted pressure.
                if p > t:
                    state = 1
            else:
                # Symmetric signal-reversal exit: mirror flip of the entry rule.
                if p < -t:
                    state = 0
            raw[i] = state

        df = pd.DataFrame(index=idx)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        ratio = np.where(np.isfinite(range_ratio), range_ratio, 1.0)
        ratio = np.clip(ratio, 0.7, 1.3)
        sv = np.where(np.isfinite(size_vol), size_vol, float(params.min_size))
        size = np.clip(sv * ratio, float(params.min_size), float(params.max_size))
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
