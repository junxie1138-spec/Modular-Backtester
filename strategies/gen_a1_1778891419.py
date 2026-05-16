from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    peak_window: int = 60
    dd_min: float = 0.03
    decay: float = 0.85
    gap_std_window: int = 40
    cap_mult: float = 4.0
    profit_target: float = 0.05
    max_hold: int = 18
    sz_min: float = 0.5
    sz_max: float = 1.0
    sz_span: float = 1.0


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    """Drawdown-gated leaky bucket of overnight down-gaps.

    While SPY trades below its rolling peak, the magnitude of every overnight
    down-gap is fed into a leaky bucket (an exponentially decaying running
    sum). When the bucket overflows a volatility-scaled capacity ceiling for
    the first time, the strategy reads it as overnight-selling capitulation
    and opens a long. Exit fires at a profit target or a hard time-stop,
    whichever comes first. Position size scales with how far the bucket ran
    past capacity at entry.
    """

    strategy_id = "gen_a1_1778891419"

    @classmethod
    def params_type(cls):
        return GapDebtParams

    @staticmethod
    def warmup_bars(params: GapDebtParams) -> int:
        return int(max(params.peak_window, params.gap_std_window)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapDebtParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        prev_close = close.shift(1)
        gap = (open_ / prev_close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        down_gap = (-gap).clip(lower=0.0)

        peak_window = int(max(params.peak_window, 1))
        peak = close.rolling(peak_window, min_periods=1).max()
        drawdown = close / peak - 1.0
        dd_depth = (-drawdown).clip(lower=0.0).fillna(0.0)

        # Leaky bucket: s[t] = decay * s[t-1] + inflow[t], implemented as a
        # rescaled EWM running mean so it stays fully vectorised and NaN-safe.
        decay = float(min(max(params.decay, 0.01), 0.99))
        alpha = 1.0 - decay
        in_dd = dd_depth >= float(params.dd_min)
        inflow = down_gap.where(in_dd, 0.0)
        bucket = inflow.ewm(alpha=alpha, adjust=False).mean() / alpha
        bucket = bucket.fillna(0.0)

        # Volatility-scaled capacity ceiling from gap dispersion.
        gap_std_window = int(max(params.gap_std_window, 2))
        gap_std = gap.rolling(gap_std_window, min_periods=gap_std_window).std()
        capacity = float(params.cap_mult) * gap_std
        capacity = capacity.where(capacity > 0.0)
        capacity_filled = capacity.fillna(np.inf)

        overflow = bucket > capacity_filled
        fresh = overflow & ~overflow.shift(1, fill_value=False)
        entry_raw = (fresh & in_dd).astype(float)

        ratio = (bucket / capacity_filled).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        span = float(max(params.sz_span, 1e-6))
        strength = ((ratio - 1.0) / span).clip(0.0, 1.0)
        sizing = float(params.sz_min) + (float(params.sz_max) - float(params.sz_min)) * strength
        sizing = sizing.clip(lower=1e-6).fillna(float(params.sz_min))

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["dd_depth"] = dd_depth
        out["bucket"] = bucket
        out["capacity"] = capacity_filled
        out["entry_raw"] = entry_raw
        out["sizing"] = sizing
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDebtParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].to_numpy(dtype=float)
        sizing_arr = indicators["sizing"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        bars_held = 0
        held_size = 1.0
        pt = float(params.profit_target)
        max_hold = int(max(params.max_hold, 1))
        sz_min = float(params.sz_min)

        for i in range(n):
            if not in_pos:
                if (
                    entry_raw[i] >= 0.5
                    and np.isfinite(close[i])
                    and close[i] > 0.0
                ):
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    held_size = sizing_arr[i]
                    if not np.isfinite(held_size) or held_size <= 0.0:
                        held_size = sz_min
                    signal[i] = 1
                    size[i] = held_size
                else:
                    signal[i] = 0
                    size[i] = 1.0
            else:
                bars_held += 1
                ret = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if ret >= pt or bars_held >= max_hold:
                    in_pos = False
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = 1
                    size[i] = held_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
