from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBreakoutParams:
    pct_window: int = 60
    pctile_threshold: float = 0.90
    atr_window: int = 20
    snr_min: float = 1.0
    profit_target: float = 0.04
    max_bars: int = 5
    target_atr: float = 0.015


class GeneratedStrategy(BaseStrategy[GapBreakoutParams]):
    strategy_id = "gen_a1_1778914144"

    @classmethod
    def params_type(cls) -> type[GapBreakoutParams]:
        return GapBreakoutParams

    @staticmethod
    def warmup_bars(params: GapBreakoutParams) -> int:
        return int(max(params.pct_window, params.atr_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapBreakoutParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        prior_close = close.shift(1)

        # Overnight gap as a fraction of prior close.
        gap = open_ / prior_close - 1.0
        abs_gap = gap.abs()

        # Percentile threshold twist: rank of today's |gap| within its own
        # recent magnitude distribution (0..1). NaN during warmup.
        pct_w = max(int(params.pct_window), 2)
        gap_rank = abs_gap.rolling(pct_w).rank(pct=True)

        # ATR-based noise floor for the signal-to-noise filter.
        hl = high - low
        hc = (high - prior_close).abs()
        lc = (low - prior_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(max(int(params.atr_window), 2)).mean()
        rel_atr = atr / prior_close
        rel_atr_safe = rel_atr.replace(0.0, np.nan)

        # SNR: gap size relative to typical relative true range.
        snr = abs_gap / rel_atr_safe

        # Mild inverse-vol sizing, clamped and NaN-safe.
        size = (params.target_atr / rel_atr_safe).clip(lower=0.5, upper=1.5)
        size = size.fillna(1.0)

        ind = pd.DataFrame(index=data.index)
        ind["gap"] = gap
        ind["gap_rank"] = gap_rank
        ind["snr"] = snr
        ind["rel_atr"] = rel_atr
        ind["size"] = size
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBreakoutParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        open_ = data["open"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        gap_rank = indicators["gap_rank"].to_numpy(dtype=float)
        snr = indicators["snr"].to_numpy(dtype=float)

        n = len(close)
        thr = float(params.pctile_threshold)
        snr_min = float(params.snr_min)
        target = float(params.profit_target)
        max_bars = max(int(params.max_bars), 1)

        # NaN-safe substitutions: warmup bars fail every comparison.
        gap_f = np.nan_to_num(gap, nan=0.0)
        rank_f = np.nan_to_num(gap_rank, nan=-1.0)
        snr_f = np.nan_to_num(snr, nan=-1.0)

        big = rank_f >= thr
        clean = snr_f >= snr_min
        # Intraday-hold confirmation: an up-gap must not retrace below its
        # open by the close; a down-gap must not rally back above its open.
        long_entry = (gap_f > 0.0) & big & clean & (close >= open_)
        short_entry = (gap_f < 0.0) & big & clean & (close <= open_)

        pos = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(n):
            if position == 0:
                if long_entry[i]:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    pos[i] = 1
                elif short_entry[i]:
                    position = -1
                    entry_price = close[i]
                    bars_held = 0
                    pos[i] = -1
                else:
                    pos[i] = 0
            else:
                bars_held += 1
                if entry_price > 0.0:
                    ret = position * (close[i] / entry_price - 1.0)
                else:
                    ret = 0.0
                # Exit at the profit target or the time-stop, first to fire.
                if ret >= target or bars_held >= max_bars:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    pos[i] = 0
                else:
                    pos[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].to_numpy(dtype=float)
        size = np.where(np.isnan(size) | (size <= 0.0), 1.0, size)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
