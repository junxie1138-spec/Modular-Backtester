from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    lookback: int = 63
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    """Drawdown-jam kinematic-wave reversal (Greenshields/LWR traffic model).

    Drawdown depth is treated as traffic density rho. Under the Greenshields
    fundamental diagram the kinematic wave speed is dq/drho = 1 - 2*rho once
    rho is normalised to [0, 1] by a rolling capacity rho_max. The wave speed
    is negative (congested, jam propagates backward) for rho > 0.5 and positive
    (free flow) for rho < 0.5. A long fires when the drawdown jam's wave speed
    crosses up through zero after sustained congestion; a short fires on the
    mirror crossing of a run-up 'anti-jam' measured above the rolling trough.
    """

    strategy_id = "gen_a1_1778892955"

    # fixed exit constants (not tunable params)
    _PROFIT_TARGET = 0.05
    _MIN_JAM_BARS = 3

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(2 * max(2, params.lookback) + 5)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        idx = close.index
        w = int(max(2, params.lookback))
        eps = 1e-9
        n = len(close)

        # --- drawdown jam (long side): density = depth below rolling peak ---
        peak = close.rolling(w, min_periods=w).max()
        peak_safe = peak.where(peak > eps, np.nan)
        depth = ((peak - close) / peak_safe).clip(lower=0.0)
        depth_max = depth.rolling(w, min_periods=w).max()
        d = depth.to_numpy()
        dm = depth_max.to_numpy()
        valid_l = np.isfinite(d) & np.isfinite(dm) & (dm > eps)
        rho_l = np.where(valid_l, np.divide(d, dm, out=np.zeros(n), where=valid_l), 0.0)
        rho_l = np.clip(rho_l, 0.0, 1.0)
        w_long = 1.0 - 2.0 * rho_l

        # --- run-up anti-jam (short side): density = extension above trough ---
        trough = close.rolling(w, min_periods=w).min()
        trough_safe = trough.where(trough > eps, np.nan)
        ext = ((close - trough) / trough_safe).clip(lower=0.0)
        ext_max = ext.rolling(w, min_periods=w).max()
        e = ext.to_numpy()
        em = ext_max.to_numpy()
        valid_s = np.isfinite(e) & np.isfinite(em) & (em > eps)
        rho_s = np.where(valid_s, np.divide(e, em, out=np.zeros(n), where=valid_s), 0.0)
        rho_s = np.clip(rho_s, 0.0, 1.0)
        w_short = 1.0 - 2.0 * rho_s

        # --- wave-speed sign-flip crossings with sustained-congestion filter ---
        k = self._MIN_JAM_BARS
        jam_l = w_long < 0.0
        free_l = w_long >= 0.0
        cross_l = np.zeros(n, dtype=bool)
        jam_s = w_short < 0.0
        free_s = w_short >= 0.0
        cross_s = np.zeros(n, dtype=bool)
        if n > k:
            cl = free_l[k:].copy()
            cs = free_s[k:].copy()
            for j in range(1, k + 1):
                cl &= jam_l[k - j:n - j]
                cs &= jam_s[k - j:n - j]
            cross_l[k:] = cl
            cross_s[k:] = cs

        return pd.DataFrame(
            {
                "w_long": w_long,
                "w_short": w_short,
                "long_entry": cross_l.astype(float),
                "short_entry": cross_s.astype(float),
            },
            index=idx,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        long_e = indicators["long_entry"].to_numpy() > 0.5
        short_e = indicators["short_entry"].to_numpy() > 0.5
        n = len(close)

        hold = int(max(1, params.hold_bars))
        pt = float(self._PROFIT_TARGET)

        raw = np.zeros(n, dtype=int)
        pos = 0
        entry_px = 0.0
        held = 0

        for i in range(n):
            px = close[i]
            if pos == 0:
                if not np.isfinite(px) or px <= 0.0:
                    raw[i] = 0
                    continue
                if long_e[i]:
                    pos = 1
                    entry_px = px
                    held = 0
                elif short_e[i]:
                    pos = -1
                    entry_px = px
                    held = 0
            else:
                held += 1
                gain = 0.0
                if entry_px > 0.0 and np.isfinite(px):
                    gain = (px - entry_px) / entry_px * pos
                if gain >= pt or held >= hold:
                    pos = 0
                    entry_px = 0.0
                    held = 0
            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
