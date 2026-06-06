from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConvalescenceParams:
    dd_window: int = 60
    infect_threshold: float = 0.03
    inf_window: int = 20
    rank_window: int = 60
    p_high: float = 0.80
    p_low: float = 0.30
    p_arm: float = 0.75
    base_size: float = 1.0
    depth_scale: float = 3.0
    depth_cap: float = 0.20


class GeneratedStrategy(BaseStrategy[ConvalescenceParams]):
    """Drawdown-recovery: buy the convalescent phase when two rolling percentile
    ranks agree, gated by a prior SI-style epidemic arm. Signal-reversal exit."""

    strategy_id = "gen_a1_1778892623"

    @classmethod
    def params_type(cls):
        return ConvalescenceParams

    def warmup_bars(self, params: ConvalescenceParams) -> int:
        return int(params.dd_window + params.inf_window + params.rank_window + 5)

    def indicators(self, data: pd.DataFrame, params: ConvalescenceParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)

        dd_w = max(2, int(params.dd_window))
        inf_w = max(2, int(params.inf_window))
        rank_w = max(2, int(params.rank_window))
        thr = abs(float(params.infect_threshold))

        # Drawdown from a rolling peak (<= 0, NaN during warmup).
        roll_max = close.rolling(dd_w, min_periods=dd_w).max()
        drawdown = close / roll_max - 1.0

        # SI 'infected' bar: drawdown deeper than the threshold.
        infected = (drawdown < -thr).astype(float)
        # Infected fraction = rolling share of stressed bars (the I curve).
        infected_fraction = infected.rolling(inf_w, min_periods=inf_w).mean()

        # Primary primitive: rolling percentile ranks of both series.
        rank_dd = drawdown.rolling(rank_w, min_periods=rank_w).rank(pct=True)
        rank_inf = infected_fraction.rolling(rank_w, min_periods=rank_w).rank(pct=True)

        out["drawdown"] = drawdown
        out["infected_fraction"] = infected_fraction
        out["rank_dd"] = rank_dd
        out["rank_inf"] = rank_inf
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ConvalescenceParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(data)

        dd = indicators["drawdown"].to_numpy(dtype=float)
        rank_dd = indicators["rank_dd"].to_numpy(dtype=float)
        rank_inf = indicators["rank_inf"].to_numpy(dtype=float)

        valid_dd = ~np.isnan(rank_dd)
        valid_inf = ~np.isnan(rank_inf)
        rd = np.where(valid_dd, rank_dd, 0.0)
        ri = np.where(valid_inf, rank_inf, 0.0)

        # Two-primitive AND: drawdown rank shallow (recovering) AND
        # infected-fraction rank low (epidemic subsiding). Both must agree.
        prim_a = valid_dd & (rd > float(params.p_high))
        prim_b = valid_inf & (ri < float(params.p_low))
        cond = prim_a & prim_b
        # Arm: the epidemic must have raged (infected-fraction rank high) first.
        arm = valid_inf & (ri > float(params.p_arm))

        base = float(params.base_size)
        cap = abs(float(params.depth_cap))
        scale = float(params.depth_scale)

        sig = np.zeros(n, dtype=int)
        size = np.full(n, base, dtype=float)

        state = 0          # 0 = flat, 1 = long
        armed = False
        trough = 0.0       # worst drawdown observed while armed
        entry_size = base

        for i in range(n):
            d = dd[i]
            if np.isnan(d):
                d = 0.0

            if arm[i]:
                armed = True
                if d < trough:
                    trough = d

            if state == 0:
                if armed and cond[i]:
                    depth = min(abs(trough), cap)
                    entry_size = base * (1.0 + scale * depth)
                    state = 1
                    armed = False
                    trough = 0.0
                    sig[i] = 1
                    size[i] = entry_size
            else:
                # Signal-reversal exit: hold long until the entry AND flips.
                if cond[i]:
                    sig[i] = 1
                    size[i] = entry_size
                else:
                    state = 0
                    sig[i] = 0

        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        signal = pd.Series(sig, index=idx).shift(1).fillna(0).astype(int)
        size_s = pd.Series(size, index=idx).shift(1).fillna(base)
        size_s = size_s.clip(lower=1e-6)

        df = pd.DataFrame({"signal": signal, "size": size_s}, index=idx)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
