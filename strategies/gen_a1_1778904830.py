from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTensionParams:
    ma_period: int = 200
    gap_ewma_span: int = 10
    gap_std_window: int = 60
    tension_entry: float = 0.8
    profit_target: float = 0.08
    time_stop_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapTensionParams]):
    strategy_id = "gen_a1_1778904830"

    @classmethod
    def params_type(cls) -> type[GapTensionParams]:
        return GapTensionParams

    def warmup_bars(self, params: GapTensionParams) -> int:
        return int(max(params.ma_period, params.gap_std_window)) + 1

    def indicators(self, data: pd.DataFrame, params: GapTensionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        close = data["close"]
        open_ = data["open"]

        # Overnight gap as a fractional return: today's open vs prior close.
        gap = open_ / close.shift(1) - 1.0

        # Decaying reservoir of recent gaps - stored 'spring tension'.
        gap_ewma = gap.ewm(span=max(int(params.gap_ewma_span), 1), adjust=False).mean()

        # Volatility of gaps over a longer window for normalization.
        gap_std = gap.rolling(int(params.gap_std_window), min_periods=int(params.gap_std_window)).std()

        # Tension: how stretched the gap reservoir is relative to typical gap noise.
        tension = gap_ewma / (gap_std + 1e-9)

        # Long-term regime filter.
        ma = close.rolling(int(params.ma_period), min_periods=int(params.ma_period)).mean()

        out["gap"] = gap
        out["tension"] = tension
        out["ma"] = ma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTensionParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        tension = indicators["tension"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        thr = float(params.tension_entry)
        target = float(params.profit_target)
        max_hold = int(params.time_stop_bars)

        pos = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            t_now = tension[i]
            t_prev = tension[i - 1] if i > 0 else np.nan
            ma_now = ma[i]
            px = close[i]

            if in_pos:
                bars_held += 1
                gain = px / entry_price - 1.0 if entry_price > 0.0 else 0.0
                # Exit: profit-target reached OR time-stop elapsed, whichever first.
                if gain >= target or bars_held >= max_hold:
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                # Entry: tension crosses up through its elastic limit while in an uptrend.
                crossed = (
                    np.isfinite(t_now)
                    and np.isfinite(t_prev)
                    and t_prev <= thr
                    and t_now > thr
                )
                regime_ok = np.isfinite(ma_now) and px > ma_now
                if crossed and regime_ok:
                    in_pos = True
                    entry_price = px
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
