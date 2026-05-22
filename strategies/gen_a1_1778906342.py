from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HysteresisSetParams:
    window: int = 12
    vol_window: int = 20
    yield_threshold: float = 1.2
    retention_threshold: float = 0.55
    hold_bars: int = 17
    base_size: float = 0.3
    max_size: float = 1.0
    min_size: float = 0.3
    conviction_cap: float = 3.0


class GeneratedStrategy(BaseStrategy[HysteresisSetParams]):
    """Plastic-hysteresis entry on close-to-close return paths.

    Builds the cumulative close-to-close return path, measures the peak
    excursion inside a rolling window versus the residual (permanent) set at
    the window end, and trades the direction of the set only when the peak
    strain clears the elastic random-walk limit and a high fraction of it was
    retained. Position is held for a fixed number of bars, sized by conviction.
    """

    strategy_id = "gen_a1_1778906342"

    @classmethod
    def params_type(cls) -> type[HysteresisSetParams]:
        return HysteresisSetParams

    @staticmethod
    def warmup_bars(params: HysteresisSetParams) -> int:
        return int(params.window + params.vol_window + 2)

    def indicators(self, data: pd.DataFrame, params: HysteresisSetParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        r = close.pct_change()

        W = int(params.window)

        # Global cumulative close-to-close return path (skipna keeps NaN at 0).
        R = r.cumsum()
        R_prev = R.shift(W)                 # path value just before window start
        R_max = R.rolling(W).max()          # most positive point inside window
        R_min = R.rolling(W).min()          # most negative point inside window

        # Displacements measured from the window start.
        peak_up = R_max - R_prev            # peak up-excursion
        peak_dn = R_min - R_prev            # peak down-excursion (negative)
        net = R - R_prev                    # residual / permanent set at window end

        # Elastic random-walk scale: per-bar vol times sqrt(window).
        sigma = r.rolling(int(params.vol_window)).std()
        elastic = (sigma * np.sqrt(float(W))).replace(0.0, np.nan)

        # Normalized peak strain per direction.
        strain_up = peak_up / elastic
        strain_dn = (-peak_dn) / elastic

        # Retention ratio: fraction of the peak excursion that became set.
        retention_up = net / peak_up.replace(0.0, np.nan)
        retention_dn = (-net) / (-peak_dn).replace(0.0, np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["strain_up"] = strain_up
        ind["strain_dn"] = strain_dn
        ind["retention_up"] = retention_up
        ind["retention_dn"] = retention_dn
        ind["peak_up"] = peak_up
        ind["peak_dn"] = peak_dn
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HysteresisSetParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        strain_up = indicators["strain_up"].to_numpy(dtype=float)
        strain_dn = indicators["strain_dn"].to_numpy(dtype=float)
        ret_up = indicators["retention_up"].to_numpy(dtype=float)
        ret_dn = indicators["retention_dn"].to_numpy(dtype=float)
        peak_up = indicators["peak_up"].to_numpy(dtype=float)
        peak_dn = indicators["peak_dn"].to_numpy(dtype=float)

        yth = float(params.yield_threshold)
        rth = float(params.retention_threshold)
        hold = max(1, int(params.hold_bars))
        cap = float(params.conviction_cap)
        base = float(params.base_size)
        smin = float(params.min_size)
        smax = float(params.max_size)

        # Per-bar entry intent and conviction-scaled size.
        entry_dir = np.zeros(n, dtype=int)
        entry_size = np.ones(n, dtype=float)

        for t in range(n):
            su = strain_up[t]
            sd = strain_dn[t]
            ru = ret_up[t]
            rd = ret_dn[t]
            pu = peak_up[t]
            pdn = peak_dn[t]

            if not (np.isfinite(su) and np.isfinite(sd)):
                continue

            # Dominant deformation = the larger absolute excursion in the window.
            mag_up = pu if np.isfinite(pu) else -1.0e9
            mag_dn = (-pdn) if np.isfinite(pdn) else -1.0e9
            up_dominant = mag_up >= mag_dn

            if up_dominant:
                if su > yth and np.isfinite(ru) and ru > rth:
                    strain_factor = min(su / yth, cap)
                    ret_factor = min(max(ru, 0.0), 1.5)
                    sz = base * strain_factor * ret_factor
                    entry_dir[t] = 1
                    entry_size[t] = float(np.clip(sz, smin, smax))
            else:
                if sd > yth and np.isfinite(rd) and rd > rth:
                    strain_factor = min(sd / yth, cap)
                    ret_factor = min(max(rd, 0.0), 1.5)
                    sz = base * strain_factor * ret_factor
                    entry_dir[t] = -1
                    entry_size[t] = float(np.clip(sz, smin, smax))

        # Fixed-bar exit: hold exactly `hold` bars, no re-entry while held.
        position = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)
        t = 0
        while t < n:
            if entry_dir[t] != 0:
                end = min(t + hold, n)
                position[t:end] = entry_dir[t]
                size_arr[t:end] = entry_size[t]
                t = end
            else:
                t += 1

        df = pd.DataFrame(index=idx)
        df["signal"] = position
        df["size"] = size_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=smin)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
