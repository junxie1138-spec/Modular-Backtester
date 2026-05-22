from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    atr_window: int = 14
    pct_lookback: int = 100
    upper_pct: float = 0.70
    lower_pct: float = 0.30
    gap_pct_threshold: float = 0.60
    atr_stop_mult: float = 2.0
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778898447"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.vol_window, params.atr_window, params.pct_lookback)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        eps = 1e-12

        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        # Decompose returns into overnight (gap) and intraday components.
        gap_ret = open_ / (prev_close + eps) - 1.0
        intraday_ret = close / (open_ + eps) - 1.0

        gap_vol = gap_ret.rolling(params.vol_window).std()
        intraday_vol = intraday_ret.rolling(params.vol_window).std()
        ratio = gap_vol / (intraday_vol + eps)

        # ATR for the fixed volatility stop and gap normalization.
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        atr_frac = atr / (close + eps)
        norm_gap = gap_ret.abs() / (atr_frac + eps)

        # Percentile-rank threshold twist: regime and magnitude gate are both
        # measured against each series' own trailing distribution.
        ratio_pct = ratio.rolling(params.pct_lookback).rank(pct=True)
        norm_gap_pct = norm_gap.rolling(params.pct_lookback).rank(pct=True)

        gap_sign = np.sign(gap_ret).fillna(0.0)

        # Regime: +1 = overnight vol dominates -> follow the gap;
        #         -1 = intraday vol dominates -> fade the gap;
        #          0 = hysteresis dead zone between the percentile bands.
        regime = pd.Series(0.0, index=data.index)
        regime = regime.where(~(ratio_pct >= params.upper_pct), 1.0)
        regime = regime.where(~(ratio_pct <= params.lower_pct), -1.0)

        direction = regime * gap_sign
        mag_gate = (norm_gap_pct >= params.gap_pct_threshold).astype(float)
        raw_entry = (direction * mag_gate).fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["raw_entry"] = raw_entry
        out["ratio_pct"] = ratio_pct
        out["norm_gap_pct"] = norm_gap_pct
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw_entry = indicators["raw_entry"].to_numpy(dtype=float)

        n = len(close)
        position = np.zeros(n, dtype=int)

        k = float(params.atr_stop_mult)
        max_hold = int(params.max_hold)

        pos = 0
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if pos != 0:
                bars_held += 1
                exit_now = False
                # Fixed volatility stop: level frozen at entry, never trailed.
                if pos == 1 and np.isfinite(close[i]) and close[i] <= stop_level:
                    exit_now = True
                elif pos == -1 and np.isfinite(close[i]) and close[i] >= stop_level:
                    exit_now = True
                # Time stop enforces the 1-2 day holding horizon.
                if bars_held >= max_hold:
                    exit_now = True
                if exit_now:
                    pos = 0
                    stop_level = 0.0
                    bars_held = 0

            if pos == 0:
                sig = raw_entry[i]
                a = atr[i]
                if sig != 0.0 and np.isfinite(a) and a > 0.0 and np.isfinite(close[i]):
                    pos = 1 if sig > 0.0 else -1
                    entry_price = close[i]
                    if pos == 1:
                        stop_level = entry_price - k * a
                    else:
                        stop_level = entry_price + k * a
                    bars_held = 0

            position[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
