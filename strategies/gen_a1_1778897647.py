from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 6
    k: float = 2.0
    band_lo: float = 0.3
    band_hi: float = 0.65
    skew_ratio: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778897647"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # rolling windows of length `window` applied to pct_change (1 leading NaN);
        # window + 2 covers the binding lookback. window default 6 -> warmup 8 (<=10).
        return int(params.window) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        n = max(int(params.window), 2)
        close = data["close"].astype(float)

        ret = close.pct_change()
        up = ret.clip(lower=0.0)
        dn = (-ret).clip(lower=0.0)

        # asymmetric volatility corridor half-widths (return semi-deviations)
        up_dev = np.sqrt((up * up).rolling(n, min_periods=n).mean())
        dn_dev = np.sqrt((dn * dn).rolling(n, min_periods=n).mean())

        mid = close.rolling(n, min_periods=n).mean()
        upper = mid * (1.0 + float(params.k) * up_dev)
        lower = mid * (1.0 - float(params.k) * dn_dev)

        width = (upper - lower).replace(0.0, np.nan)
        rp = (close - lower) / width  # relative position of close inside the corridor

        out = pd.DataFrame(index=data.index)
        out["rp"] = rp
        out["up_dev"] = up_dev
        out["dn_dev"] = dn_dev
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        rp = indicators["rp"].to_numpy(dtype=float)
        up_dev = indicators["up_dev"].to_numpy(dtype=float)
        dn_dev = indicators["dn_dev"].to_numpy(dtype=float)

        n = len(data)
        sig = np.zeros(n, dtype=np.int64)

        band_lo = float(params.band_lo)
        band_hi = float(params.band_hi)
        if band_hi <= band_lo:
            band_hi = band_lo + 0.1
        skew = float(params.skew_ratio)

        state = 0
        for i in range(n):
            r = rp[i]
            if not np.isfinite(r):
                state = 0
                sig[i] = 0
                continue
            if state == 0:
                ud = up_dev[i]
                dd = dn_dev[i]
                fear = (
                    np.isfinite(ud)
                    and np.isfinite(dd)
                    and dd >= skew * ud
                )
                # entry: close in the low region of the asymmetric vol corridor
                # while downside semi-deviation dominates
                if r <= band_lo and fear:
                    state = 1
            else:
                # signal-reversal exit: exit only when the entry condition flips,
                # i.e. relative position crosses to the opposite (high) extreme
                if r >= band_hi:
                    state = 0
            sig[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
