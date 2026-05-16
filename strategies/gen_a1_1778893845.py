from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockwaveParams:
    comp_win: int = 10
    pct_win: int = 120
    comp_pct: float = 0.20
    release_win: int = 3
    breakout_k: float = 1.0
    hold_bars: int = 2
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[RangeShockwaveParams]):
    """Range-compression shockwave: enter long when return dispersion is jammed
    into a low percentile AND a return escapes beyond the compressed band.
    Fixed-bar exit exactly hold_bars after entry."""

    strategy_id = "gen_a1_1778893845"

    @classmethod
    def params_type(cls) -> type[RangeShockwaveParams]:
        return RangeShockwaveParams

    @staticmethod
    def warmup_bars(params: RangeShockwaveParams) -> int:
        # pct_win ranks over disp, disp needs comp_win returns, returns need +1.
        return int(params.pct_win + params.comp_win + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeShockwaveParams) -> pd.DataFrame:
        p = params
        close = data["close"].astype(float)

        # Primary primitive: close-to-close returns.
        ret = close.pct_change()

        # Primitive A - range compression: rolling dispersion of returns,
        # ranked as a percentile of its own trailing history. Low rank = jam.
        disp = ret.rolling(p.comp_win).std()
        disp_rank = disp.rolling(p.pct_win).rank(pct=True)
        compressed = disp_rank <= p.comp_pct

        # Primitive B - directional release (shockwave front): the latest
        # return escapes beyond k times the compressed band, with positive
        # recent cumulative return so the wave propagates upward.
        cum_release = ret.rolling(p.release_win).sum()
        escape = ret > (p.breakout_k * disp)
        release_up = (cum_release > 0.0) & escape

        # Two-primitive AND: both the coil and the release must agree.
        raw_long = (compressed & release_up).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["disp"] = disp
        out["disp_rank"] = disp_rank
        out["cum_release"] = cum_release
        out["raw_long"] = raw_long.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeShockwaveParams,
    ) -> SignalFrame:
        p = params
        n = len(data)
        hold = max(1, int(p.hold_bars))
        base = float(p.base_size)
        comp_pct = float(p.comp_pct) if p.comp_pct > 0.0 else 1.0

        raw_long = indicators["raw_long"].fillna(0.0).to_numpy() > 0.5
        disp_rank = indicators["disp_rank"].to_numpy()

        signal = np.zeros(n, dtype=int)
        size = np.full(n, base, dtype=float)

        # Fixed-bar exit: hold exactly `hold` bars after entry, then flat.
        # Refractory while in position; re-entry allowed on the exit bar.
        exit_idx = -1
        for i in range(n):
            if i < exit_idx:
                signal[i] = 1
                continue
            if raw_long[i]:
                dr = disp_rank[i]
                if np.isnan(dr):
                    depth = 0.0
                else:
                    depth = (comp_pct - float(dr)) / comp_pct
                    depth = min(1.0, max(0.0, depth))
                scaled = base * (1.0 + depth)
                exit_idx = i + hold
                end = min(exit_idx, n)
                for j in range(i, end):
                    signal[j] = 1
                    size[j] = scaled

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N's close, fill on N+1.
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = (
            pd.Series(size, index=data.index).shift(1).fillna(base).astype(float)
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
