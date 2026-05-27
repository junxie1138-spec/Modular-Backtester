from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionParams:
    short_vol_window: int = 5
    long_vol_window: int = 60
    rel_pos_window: int = 60
    percentile_lookback: int = 252
    vol_ratio_pctile_threshold: float = 0.85
    rel_pos_pctile_threshold: float = 0.15
    hold_bars: int = 18
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[SpringTensionParams]):
    strategy_id = "gen_a1_1779653811"

    @classmethod
    def params_type(cls):
        return SpringTensionParams

    @classmethod
    def warmup_bars(cls, params: SpringTensionParams) -> int:
        return int(params.long_vol_window + params.percentile_lookback + 2)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: SpringTensionParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        returns = close.pct_change()

        short_w = int(params.short_vol_window)
        long_w = int(params.long_vol_window)
        rp_w = int(params.rel_pos_window)
        pct_w = int(params.percentile_lookback)

        short_vol = returns.rolling(short_w, min_periods=short_w).std()
        long_vol = returns.rolling(long_w, min_periods=long_w).std()
        long_vol_safe = long_vol.replace(0.0, np.nan)
        vol_ratio = short_vol / long_vol_safe

        roll_max = high.rolling(rp_w, min_periods=rp_w).max()
        roll_min = low.rolling(rp_w, min_periods=rp_w).min()
        rng = (roll_max - roll_min).replace(0.0, np.nan)
        rel_pos = (close - roll_min) / rng

        vol_ratio_pctile = vol_ratio.rolling(pct_w, min_periods=pct_w).rank(pct=True)
        rel_pos_pctile = rel_pos.rolling(pct_w, min_periods=pct_w).rank(pct=True)

        return pd.DataFrame(
            {
                "vol_ratio": vol_ratio,
                "rel_pos": rel_pos,
                "vol_ratio_pctile": vol_ratio_pctile,
                "rel_pos_pctile": rel_pos_pctile,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionParams,
    ) -> SignalFrame:
        vrp = indicators["vol_ratio_pctile"].to_numpy(dtype=float)
        rpp = indicators["rel_pos_pctile"].to_numpy(dtype=float)

        valid = ~(np.isnan(vrp) | np.isnan(rpp))
        vr_thr = float(params.vol_ratio_pctile_threshold)
        rp_thr = float(params.rel_pos_pctile_threshold)
        entry = valid & (vrp >= vr_thr) & (rpp <= rp_thr)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        hold = max(1, int(params.hold_bars))
        bars_in_trade = 0
        in_trade = False
        for i in range(n):
            if in_trade:
                raw[i] = 1
                bars_in_trade += 1
                if bars_in_trade >= hold:
                    in_trade = False
                    bars_in_trade = 0
            elif entry[i]:
                in_trade = True
                bars_in_trade = 1
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size_val = float(params.size) if float(params.size) > 0.0 else 1.0
        df["size"] = size_val
        return SignalFrame(data=df, signal_column="signal", size_column="size")
