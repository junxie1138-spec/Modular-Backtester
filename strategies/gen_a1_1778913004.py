from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilEfficiencyParams:
    comp_window: int = 20
    comp_pct_window: int = 100
    comp_pct_threshold: float = 0.30
    er_threshold: float = 0.35
    atr_window: int = 14
    atr_k: float = 2.0
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CoilEfficiencyParams]):
    strategy_id = "gen_a1_1778913004"

    @classmethod
    def params_type(cls):
        return CoilEfficiencyParams

    @staticmethod
    def warmup_bars(params: CoilEfficiencyParams) -> int:
        return int(
            max(
                params.comp_window + params.comp_pct_window + 1,
                params.atr_window + 1,
            )
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: CoilEfficiencyParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Primary primitive: close-to-close returns.
        r = close.pct_change(fill_method=None)

        # Primitive A - range/dispersion compression: rolling std of returns
        # ranked as a percentile against its own recent history. A low rank
        # means the return convoy has coiled into a tight, quiet range.
        ret_std = r.rolling(params.comp_window).std()
        ret_std_rank = ret_std.rolling(params.comp_pct_window).rank(pct=True)
        compressed = (ret_std_rank < params.comp_pct_threshold).fillna(False)

        # Primitive B - signal-to-noise efficiency of the return path.
        # net displacement / gross path length; high value with positive net
        # means the small returns are coherently drifting upward (low noise).
        net = r.rolling(params.comp_window).sum()
        gross = r.abs().rolling(params.comp_window).sum()
        er = net.abs() / gross.replace(0.0, np.nan)
        snr_ok = ((er > params.er_threshold) & (net > 0.0)).fillna(False)

        # Hard twist: two-primitive AND - both must agree to enter.
        entry_signal = compressed & snr_ok

        # ATR for the fixed volatility-stop exit.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["er"] = er
        out["ret_std_rank"] = ret_std_rank
        out["entry_signal"] = entry_signal.astype(bool)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilEfficiencyParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry_signal"].to_numpy()

        pos = np.zeros(n, dtype=int)
        in_pos = False
        stop = 0.0
        bars_held = 0
        k = float(params.atr_k)
        max_hold = int(params.max_hold_bars)

        # Bar-indexed loop: the fixed volatility-stop and time cap are
        # path-dependent and have no clean vectorised equivalent.
        for i in range(n):
            if in_pos:
                bars_held += 1
                # Fixed (non-trailing) ATR vol-stop: stop level was frozen at
                # entry. Also cap the hold to the 1-2 day target horizon.
                if close[i] <= stop or bars_held >= max_hold:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                if bool(entry[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    stop = close[i] - k * atr[i]
                    bars_held = 0
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
