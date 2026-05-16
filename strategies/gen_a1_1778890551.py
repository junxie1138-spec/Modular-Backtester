from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AutocorrRegimeParams:
    autocorr_window: int = 30
    percentile_lookback: int = 120
    direction_horizon: int = 3
    hold_bars: int = 4
    q_high: float = 0.80
    q_low: float = 0.20


class GeneratedStrategy(BaseStrategy[AutocorrRegimeParams]):
    strategy_id = "gen_a1_1778890551"

    @classmethod
    def params_type(cls) -> type[AutocorrRegimeParams]:
        return AutocorrRegimeParams

    @staticmethod
    def warmup_bars(params: AutocorrRegimeParams) -> int:
        return int(
            max(int(params.autocorr_window), 3)
            + max(int(params.percentile_lookback), 5)
            + max(int(params.direction_horizon), 1)
            + 2
        )

    def indicators(self, data: pd.DataFrame, params: AutocorrRegimeParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        w = max(int(params.autocorr_window), 3)
        p = max(int(params.percentile_lookback), 5)
        k = max(int(params.direction_horizon), 1)

        # Lag-1 autocorrelation of close-to-close returns: the persistence coefficient.
        rho = ret.rolling(w).corr(ret.shift(1))
        rho = rho.replace([np.inf, -np.inf], np.nan)

        # Percentile-threshold twist: compare rho to rolling quantiles of its own history.
        q_hi = float(min(max(params.q_high, 0.50), 0.99))
        q_lo = float(min(max(params.q_low, 0.01), 0.50))
        rho_high = rho.rolling(p).quantile(q_hi)
        rho_low = rho.rolling(p).quantile(q_lo)

        # k-day close-to-close return: the direction the regime points along.
        mom = close.pct_change(k)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["rho"] = rho
        out["rho_high"] = rho_high
        out["rho_low"] = rho_low
        out["mom"] = mom
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AutocorrRegimeParams,
    ) -> SignalFrame:
        n = len(data)

        rho = indicators["rho"].to_numpy(dtype=float)
        rho_high = indicators["rho_high"].to_numpy(dtype=float)
        rho_low = indicators["rho_low"].to_numpy(dtype=float)
        mom = indicators["mom"].to_numpy(dtype=float)

        mom_sign = np.sign(np.nan_to_num(mom, nan=0.0)).astype(int)

        valid = np.isfinite(rho)
        persist = valid & np.isfinite(rho_high) & (rho >= rho_high)
        antipersist = valid & np.isfinite(rho_low) & (rho <= rho_low)

        # Anti-persistence regime fades the recent return; persistence regime rides it.
        # Assign anti-persist first so persistence wins any degenerate overlap.
        entry = np.zeros(n, dtype=int)
        entry[antipersist] = -mom_sign[antipersist]
        entry[persist] = mom_sign[persist]

        # Fixed-bar exit: hold exactly `hold` bars after each entry, then go flat.
        hold = max(int(params.hold_bars), 1)
        signal = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            d = entry[i]
            if d != 0:
                end = min(i + hold, n)
                signal[i:end] = d
                i += hold
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0

        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
