from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideRecoveryParams:
    peak_window: int = 40
    range_window: int = 20
    dd_threshold: float = 0.05
    du_threshold: float = 0.05
    contraction_thr: float = 0.85
    clv_high: float = 0.60
    hold_bars: int = 2
    min_size: float = 0.40
    max_size: float = 1.00
    depth_cap: float = 3.0


class GeneratedStrategy(BaseStrategy[TideRecoveryParams]):
    strategy_id = "gen_a2_1779147217"

    @classmethod
    def params_type(cls) -> type[TideRecoveryParams]:
        return TideRecoveryParams

    @staticmethod
    def warmup_bars(params: TideRecoveryParams) -> int:
        return int(max(params.peak_window, params.range_window)) + 2

    def indicators(self, data: pd.DataFrame, params: TideRecoveryParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        rng = (high - low)
        avg_range = rng.rolling(params.range_window).mean()
        range_ratio = rng / avg_range.replace(0.0, np.nan)

        roll_max = close.rolling(params.peak_window).max()
        roll_min = close.rolling(params.peak_window).min()
        drawdown = close / roll_max.replace(0.0, np.nan) - 1.0
        drawup = close / roll_min.replace(0.0, np.nan) - 1.0

        clv = (close - low) / rng.replace(0.0, np.nan)
        clv = clv.fillna(0.5)

        out = pd.DataFrame(index=data.index)
        out["range_ratio"] = range_ratio
        out["drawdown"] = drawdown
        out["drawup"] = drawup
        out["clv"] = clv
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideRecoveryParams,
    ) -> SignalFrame:
        n = len(data)

        rr = indicators["range_ratio"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        du = indicators["drawup"].to_numpy(dtype=float)
        clv = indicators["clv"].to_numpy(dtype=float)

        valid_rr = np.isfinite(rr)
        valid_dd = np.isfinite(dd)
        valid_du = np.isfinite(du)
        valid_clv = np.isfinite(clv)

        entry_long = (
            valid_rr & valid_dd & valid_clv
            & (dd <= -params.dd_threshold)
            & (rr <= params.contraction_thr)
            & (clv >= params.clv_high)
        )
        entry_short = (
            valid_rr & valid_du & valid_clv
            & (du >= params.du_threshold)
            & (rr <= params.contraction_thr)
            & (clv <= 1.0 - params.clv_high)
        )

        dd_thr = params.dd_threshold if params.dd_threshold > 1e-9 else 1e-9
        du_thr = params.du_threshold if params.du_threshold > 1e-9 else 1e-9
        cap = params.depth_cap if params.depth_cap > 1.0 else 1.0001
        contr_thr = params.contraction_thr if params.contraction_thr > 1e-9 else 1e-9

        dd_depth = np.clip(np.abs(np.nan_to_num(dd)) / dd_thr, 1.0, cap)
        du_depth = np.clip(np.abs(np.nan_to_num(du)) / du_thr, 1.0, cap)
        depth_long_n = (dd_depth - 1.0) / (cap - 1.0)
        depth_short_n = (du_depth - 1.0) / (cap - 1.0)

        contr = np.clip((contr_thr - np.nan_to_num(rr, nan=contr_thr)) / contr_thr, 0.0, 1.0)

        span = params.max_size - params.min_size
        conv_long = np.clip(0.5 * depth_long_n + 0.5 * contr, 0.0, 1.0)
        conv_short = np.clip(0.5 * depth_short_n + 0.5 * contr, 0.0, 1.0)
        size_long = params.min_size + span * conv_long
        size_short = params.min_size + span * conv_short

        sig = np.zeros(n, dtype=int)
        sz = np.full(n, params.min_size, dtype=float)
        hold = max(1, int(params.hold_bars))

        i = 0
        while i < n:
            if entry_long[i]:
                end = min(i + hold, n)
                sig[i:end] = 1
                sz[i:end] = max(float(size_long[i]), 0.01)
                i = end
            elif entry_short[i]:
                end = min(i + hold, n)
                sig[i:end] = -1
                sz[i:end] = max(float(size_short[i]), 0.01)
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = np.where(np.isfinite(sz) & (sz > 0.0), sz, params.min_size)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
