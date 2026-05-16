from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_len: int = 50
    zscore_len: int = 60
    z_entry: float = 0.5
    adopt_len: int = 30
    adopt_cap: float = 0.75
    growth_lookback: int = 5
    hold_bars: int = 18
    vol_len: int = 20
    target_vol: float = 0.02
    size_min: float = 0.5
    size_max: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Trend-strength entry: distance-from-MA z-score AND unsaturated, rising
    epidemic-style trend adoption must both agree. Fixed-bar exit."""

    strategy_id = "gen_a1_1778915310"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        tail = max(
            params.zscore_len,
            params.adopt_len + params.growth_lookback,
            params.vol_len + 1,
        )
        return int(params.ma_len + tail + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        ma_len = max(2, int(params.ma_len))
        z_len = max(2, int(params.zscore_len))
        a_len = max(2, int(params.adopt_len))
        g_lb = max(1, int(params.growth_lookback))
        v_len = max(2, int(params.vol_len))

        ma = close.rolling(ma_len, min_periods=ma_len).mean()
        distance = close - ma

        dist_mean = distance.rolling(z_len, min_periods=z_len).mean()
        dist_std = distance.rolling(z_len, min_periods=z_len).std()
        dist_std = dist_std.replace(0.0, np.nan)
        z = (distance - dist_mean) / dist_std

        above = (close > ma).astype(float)
        adopt = above.rolling(a_len, min_periods=a_len).mean()
        susceptible = 1.0 - adopt
        foi = adopt * susceptible
        adopt_growth = adopt.diff(g_lb)

        ret = close.pct_change()
        vol = ret.rolling(v_len, min_periods=v_len).std()

        ind["ma"] = ma
        ind["distance"] = distance
        ind["z"] = z
        ind["adopt"] = adopt
        ind["susceptible"] = susceptible
        ind["foi"] = foi
        ind["adopt_growth"] = adopt_growth
        ind["vol"] = vol
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        z = indicators["z"]
        adopt = indicators["adopt"]
        adopt_growth = indicators["adopt_growth"]
        vol = indicators["vol"]

        # Primitive A: price meaningfully stretched above its MA.
        cond_a = (z > float(params.z_entry)).fillna(False)
        # Primitive B: epidemic adoption still spreading AND not yet saturated.
        cond_b = (
            (adopt_growth > 0.0) & (adopt < float(params.adopt_cap))
        ).fillna(False)
        # Two-primitive AND: both must agree.
        entry = (cond_a & cond_b).to_numpy()

        hold = max(1, int(params.hold_bars))
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        held = 0
        for i in range(n):
            if in_pos:
                raw[i] = 1
                held += 1
                if held >= hold:
                    in_pos = False
                    held = 0
            elif entry[i]:
                in_pos = True
                held = 1
                raw[i] = 1

        df = pd.DataFrame(index=idx)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size_min = float(params.size_min)
        size_max = float(params.size_max)
        if size_max < size_min:
            size_max = size_min
        size = (float(params.target_vol) / vol).clip(size_min, size_max)
        size = size.fillna(size_min)
        size = size.where(size > 0.0, size_min)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
