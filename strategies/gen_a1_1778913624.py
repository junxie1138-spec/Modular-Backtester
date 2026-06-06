from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_window: int = 6
    entry_threshold: float = 0.20
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Trend-strength via upside/downside RMS semi-deviation asymmetry.

    The trend signal is the normalized imbalance between the root-mean-square
    of positive returns and the root-mean-square of negative returns over a
    short window. A signal-to-noise filter (entry_threshold) discards bars
    where the asymmetry is too small to be distinguishable from noise. Exit
    is a strict fixed-bar exit: positions are closed exactly hold_bars after
    entry with no signal-based exit.
    """

    strategy_id = "gen_a1_1778913624"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # pct_change consumes 1 bar; the rolling RMS window consumes vol_window.
        return int(params.vol_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = max(2, int(params.vol_window))
        eps = 1e-12

        close = data["close"].astype(float)
        ret = close.pct_change()

        up = ret.clip(lower=0.0)
        down = (-ret).clip(lower=0.0)

        # One-sided root-mean-square dispersion (upside / downside semi-deviation).
        up_rms = np.sqrt((up * up).rolling(w).mean())
        down_rms = np.sqrt((down * down).rolling(w).mean())
        total_vol = up_rms + down_rms

        # Normalized volatility asymmetry in [-1, 1]: the trend-strength gauge.
        asym = (up_rms - down_rms) / (total_vol + eps)

        # Windowed cumulative return used only as a same-sign trend confirmation.
        win_ret = ret.rolling(w).sum()

        ind = pd.DataFrame(index=data.index)
        ind["up_rms"] = up_rms.fillna(0.0)
        ind["down_rms"] = down_rms.fillna(0.0)
        ind["total_vol"] = total_vol.fillna(0.0)
        ind["asym"] = asym.fillna(0.0)
        ind["win_ret"] = win_ret.fillna(0.0)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        asym = indicators["asym"].to_numpy(dtype=float)
        win_ret = indicators["win_ret"].to_numpy(dtype=float)

        thr = float(params.entry_threshold)
        hold = max(1, int(params.hold_bars))

        # Signal-to-noise filter: only act when the asymmetry clears the
        # threshold AND the windowed return confirms the same direction.
        long_mask = (asym > thr) & (win_ret > 0.0)
        short_mask = (asym < -thr) & (win_ret < 0.0)
        desired = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int64)

        # Fixed-bar exit: hold exactly `hold` bars after entry, then go flat.
        pos = np.zeros(n, dtype=np.int64)
        cur = 0
        held = 0
        for i in range(n):
            if cur != 0:
                held += 1
                if held >= hold:
                    cur = 0
                    held = 0
            if cur == 0 and desired[i] != 0:
                cur = int(desired[i])
                held = 0
            pos[i] = cur

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
