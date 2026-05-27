from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockwaveParams:
    rank_window: int = 120
    accel_smooth: int = 5
    dom_smooth: int = 5
    atr_window: int = 14
    atr_k: float = 2.5
    rank_threshold: float = 0.80
    max_hold_bars: int = 17


class GeneratedStrategy(BaseStrategy[GapShockwaveParams]):
    strategy_id = "gen_a1_1779885532"

    @classmethod
    def params_type(cls):
        return GapShockwaveParams

    @classmethod
    def warmup_bars(cls, params: GapShockwaveParams) -> int:
        base = max(
            params.rank_window,
            params.atr_window,
            params.accel_smooth,
            params.dom_smooth,
        )
        return int(base + max(params.accel_smooth, params.dom_smooth) + 5)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GapShockwaveParams) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        # Overnight gap return (signed)
        gap_ret = (open_ - prev_close) / prev_close.replace(0, np.nan)

        # Primitive A: rolling rank of smoothed gap ACCELERATION (second derivative)
        gap_accel = gap_ret.diff()
        gap_accel_smooth = gap_accel.rolling(
            params.accel_smooth, min_periods=1
        ).mean()

        min_p = max(2, params.rank_window // 2)
        accel_rank = gap_accel_smooth.rolling(
            params.rank_window, min_periods=min_p
        ).rank(pct=True)

        # True range and ATR
        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=1).mean()

        # Primitive B: rolling rank of signed gap-share-of-range
        gap_abs = (open_ - prev_close).abs()
        gap_sign = pd.Series(np.sign(open_ - prev_close), index=data.index)
        safe_tr = true_range.replace(0, np.nan)
        gap_share = gap_abs / safe_tr
        signed_gap_share = gap_sign * gap_share
        dom_smooth = signed_gap_share.rolling(
            params.dom_smooth, min_periods=1
        ).mean()
        dom_rank = dom_smooth.rolling(
            params.rank_window, min_periods=min_p
        ).rank(pct=True)

        return pd.DataFrame(
            {
                "gap_ret": gap_ret,
                "accel_rank": accel_rank,
                "dom_rank": dom_rank,
                "atr": atr,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShockwaveParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        accel_rank = indicators["accel_rank"].to_numpy(dtype=float)
        dom_rank = indicators["dom_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_bar = -1
        stop_level = np.nan

        thr = float(params.rank_threshold)
        k = float(params.atr_k)
        max_hold = int(params.max_hold_bars)

        for i in range(n):
            if in_pos:
                bars_held = i - entry_bar
                hit_stop = (not np.isnan(stop_level)) and (close[i] < stop_level)
                time_exit = bars_held >= max_hold
                if hit_stop or time_exit:
                    in_pos = False
                    entry_bar = -1
                    stop_level = np.nan
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                a = accel_rank[i]
                d = dom_rank[i]
                if (
                    not np.isnan(a)
                    and not np.isnan(d)
                    and a >= thr
                    and d >= thr
                ):
                    in_pos = True
                    entry_bar = i
                    if not np.isnan(atr[i]) and not np.isnan(close[i]):
                        stop_level = close[i] - k * atr[i]
                    else:
                        stop_level = np.nan
                    raw[i] = 1
                else:
                    raw[i] = 0

        sig = pd.Series(raw, index=data.index)
        sig = sig.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index)
        df = pd.DataFrame({"signal": sig, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
