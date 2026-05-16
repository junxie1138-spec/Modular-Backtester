from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_win: int = 3
    slow_win: int = 9
    compression_ratio: float = 0.70
    release_mult: float = 1.0
    stall_mult: float = 0.80
    hold_bars: int = 4
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778910674"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # rolling std of pct_change over slow_win needs slow_win + 1 bars
        return int(params.slow_win) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        r = close.pct_change()

        fast_win = max(2, int(params.fast_win))
        slow_win = max(fast_win + 1, int(params.slow_win))

        fast_disp = r.rolling(fast_win).std()
        slow_disp = r.rolling(slow_win).std()

        # compression: recent return dispersion collapsed below its baseline
        comp_ratio = fast_disp / slow_disp.replace(0.0, np.nan)

        # stall: net drift over the coil window is near zero
        cumret_slow = close.pct_change(slow_win)
        stall_thresh = params.stall_mult * slow_disp * np.sqrt(float(slow_win))

        # release: a single close-to-close return breaking above the band
        release_thresh = params.release_mult * slow_disp

        out = pd.DataFrame(index=data.index)
        out["r"] = r
        out["fast_disp"] = fast_disp
        out["slow_disp"] = slow_disp
        out["comp_ratio"] = comp_ratio
        out["cumret_slow"] = cumret_slow
        out["stall_thresh"] = stall_thresh
        out["release_thresh"] = release_thresh
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        idx = data.index

        r = indicators["r"].to_numpy(dtype=float)
        comp_ratio = indicators["comp_ratio"].to_numpy(dtype=float)
        cumret = indicators["cumret_slow"].to_numpy(dtype=float)
        stall_thresh = indicators["stall_thresh"].to_numpy(dtype=float)
        release_thresh = indicators["release_thresh"].to_numpy(dtype=float)

        warmup = GeneratedStrategy.warmup_bars(params)
        hold = max(1, int(params.hold_bars))

        # NaN comparisons evaluate False, so warmup bars are naturally excluded
        compression = comp_ratio < float(params.compression_ratio)
        stall = np.abs(cumret) < stall_thresh
        release = r > release_thresh
        entry = compression & stall & release

        raw = np.zeros(n, dtype=int)
        i = warmup
        while i < n:
            if bool(entry[i]):
                # fixed-bar exit: hold exactly `hold` bars, no signal-based exit
                for j in range(i, min(i + hold, n)):
                    raw[j] = 1
                i += hold
            else:
                i += 1

        df = pd.DataFrame(index=idx)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
