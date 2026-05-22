from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnEnergyParams:
    ret_window: int = 20
    entry_threshold: float = 0.35
    reset_threshold: float = 0.15
    atr_window: int = 14
    atr_mult: float = 3.0
    max_hold_bars: int = 20
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ReturnEnergyParams]):
    """Trend-strength via return-energy polarization with two-bar confirmation
    and a fixed (non-trailing) ATR volatility stop."""

    strategy_id = "gen_a1_1778907024"

    @classmethod
    def params_type(cls) -> type[ReturnEnergyParams]:
        return ReturnEnergyParams

    @staticmethod
    def warmup_bars(params: ReturnEnergyParams) -> int:
        # ret_window operates on pct_change (one extra bar); atr uses prev close.
        return int(max(params.ret_window, params.atr_window)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ReturnEnergyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ret = close.pct_change()
        up_mag = ret.clip(lower=0.0)
        dn_mag = (-ret).clip(lower=0.0)

        w = max(int(params.ret_window), 1)
        e_up = up_mag.rolling(w, min_periods=w).sum()
        e_dn = dn_mag.rolling(w, min_periods=w).sum()
        denom = (e_up + e_dn).replace(0.0, np.nan)
        # Signed polarization in [-1, 1]: +1 = all return energy upward.
        polarization = ((e_up - e_dn) / denom).fillna(0.0)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        aw = max(int(params.atr_window), 1)
        atr = true_range.rolling(aw, min_periods=aw).mean()

        out = pd.DataFrame(index=data.index)
        out["polarization"] = polarization
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnEnergyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        pol = indicators["polarization"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        entry = float(params.entry_threshold)
        reset = float(params.reset_threshold)
        k = float(params.atr_mult)
        max_hold = max(int(params.max_hold_bars), 1)

        sig = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        held = 0
        can_arm = True  # hysteresis gate: blocks re-entry until polarization neutralizes

        for i in range(n):
            p = pol[i]
            c = close[i]
            a = atr[i]

            if pos == 0:
                # Hysteresis re-arm: polarization must collapse back near neutral.
                if not can_arm and abs(p) <= reset:
                    can_arm = True

                if can_arm and i >= 1 and not np.isnan(a) and a > 0.0:
                    p_prev = pol[i - 1]
                    long_conf = (p >= entry) and (p_prev >= entry)
                    short_conf = (p <= -entry) and (p_prev <= -entry)
                    if long_conf:
                        pos = 1
                        entry_price = c
                        entry_atr = a
                        held = 0
                    elif short_conf:
                        pos = -1
                        entry_price = c
                        entry_atr = a
                        held = 0
                sig[i] = pos
            else:
                held += 1
                exit_now = False
                # Fixed volatility stop: stop level frozen at entry-bar ATR.
                if pos == 1:
                    if c < entry_price - k * entry_atr:
                        exit_now = True
                else:
                    if c > entry_price + k * entry_atr:
                        exit_now = True
                if held >= max_hold:
                    exit_now = True
                if exit_now:
                    pos = 0
                    can_arm = False  # require polarization reset before next entry
                    sig[i] = 0
                else:
                    sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(max(params.base_size, 1e-9))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
