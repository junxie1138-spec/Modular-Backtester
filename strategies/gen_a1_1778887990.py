from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoilParams:
    base_win: int = 100
    dd_win: int = 40
    recoil_win: int = 5
    dd_thresh: float = 0.05
    deform_thresh: float = 1.5
    recoil_thresh: float = 1.1
    hold_bars: int = 17
    dd_cap: float = 0.20
    deform_span: float = 1.5
    size_min: float = 0.35
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[RangeRecoilParams]):
    strategy_id = "gen_a1_1778887990"

    @classmethod
    def params_type(cls):
        return RangeRecoilParams

    def warmup_bars(self, params: RangeRecoilParams) -> int:
        return int(params.base_win) + int(params.dd_win) + 1

    def indicators(self, data: pd.DataFrame, params: RangeRecoilParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"].replace(0.0, np.nan)

        base_win = max(2, int(params.base_win))
        dd_win = max(2, int(params.dd_win))
        recoil_win = max(2, int(params.recoil_win))

        rng = (high - low) / close

        rng_base = rng.rolling(base_win, min_periods=base_win).median()
        rng_short = rng.rolling(recoil_win, min_periods=recoil_win).mean()
        roll_max = close.rolling(dd_win, min_periods=dd_win).max()

        dd = close / roll_max - 1.0

        rng_base_safe = rng_base.replace(0.0, np.nan)
        deform = rng / rng_base_safe
        deform_peak = deform.rolling(dd_win, min_periods=1).max()
        recoil_ratio = rng_short / rng_base_safe

        dd_cap = float(params.dd_cap) if params.dd_cap > 0.0 else 0.20
        deform_span = float(params.deform_span) if params.deform_span > 0.0 else 1.0
        recoil_thresh = float(params.recoil_thresh) if params.recoil_thresh > 0.0 else 1.0

        depth_score = (-dd / dd_cap).clip(lower=0.0, upper=1.0)
        deform_score = ((deform_peak - params.deform_thresh) / deform_span).clip(lower=0.0, upper=1.0)
        recoil_score = ((recoil_thresh - recoil_ratio) / recoil_thresh).clip(lower=0.0, upper=1.0)
        strength = (depth_score * deform_score * recoil_score) ** (1.0 / 3.0)

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["deform_peak"] = deform_peak
        out["recoil_ratio"] = recoil_ratio
        out["strength"] = strength
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRecoilParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        dd = indicators["dd"].to_numpy(dtype=np.float64)
        deform_peak = indicators["deform_peak"].to_numpy(dtype=np.float64)
        recoil_ratio = indicators["recoil_ratio"].to_numpy(dtype=np.float64)
        strength = indicators["strength"].to_numpy(dtype=np.float64)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        hold = max(1, int(params.hold_bars))
        dd_thresh = float(params.dd_thresh)
        deform_thresh = float(params.deform_thresh)
        recoil_thresh = float(params.recoil_thresh)
        size_min = float(params.size_min)
        size_max = float(params.size_max)
        size_span = size_max - size_min

        i = 0
        while i < n:
            entry = (
                np.isfinite(dd[i])
                and np.isfinite(deform_peak[i])
                and np.isfinite(recoil_ratio[i])
                and dd[i] <= -dd_thresh
                and deform_peak[i] >= deform_thresh
                and recoil_ratio[i] <= recoil_thresh
            )
            if entry:
                s = strength[i]
                if not np.isfinite(s):
                    s = 0.0
                pos_size = size_min + s * size_span
                if pos_size <= 0.0:
                    pos_size = size_min if size_min > 0.0 else 0.1
                end = min(n, i + hold)
                signal[i:end] = 1
                size[i:min(n, end + 1)] = pos_size
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
