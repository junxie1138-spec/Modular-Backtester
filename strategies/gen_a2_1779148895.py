from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAccelParams:
    roc_len: int = 4
    hold_bars: int = 4
    tdom_lo: int = 1
    tdom_hi: int = 10
    sma_len: int = 8
    accel_eps: float = 0.0


class GeneratedStrategy(BaseStrategy[SeasonalAccelParams]):
    strategy_id = "gen_a2_1779148895"

    @classmethod
    def params_type(cls) -> type[SeasonalAccelParams]:
        return SeasonalAccelParams

    @staticmethod
    def warmup_bars(params: SeasonalAccelParams) -> int:
        # ROC of length roc_len then one .diff() -> roc_len + 1; +1 safety.
        # sma_len uses min_periods=1 so it never dominates, but guard anyway.
        return int(max(int(params.roc_len) + 2, int(params.sma_len)))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalAccelParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)

        roc_len = max(int(params.roc_len), 1)
        roc = close.pct_change(roc_len)
        accel = roc.diff()

        out["roc"] = roc
        out["accel"] = accel
        out["accel_prev"] = accel.shift(1)

        sma_len = max(int(params.sma_len), 1)
        sma = close.rolling(sma_len, min_periods=1).mean()
        out["trend_ok"] = (close > sma).astype(float)

        # Tidal phase: ordinal trading-day-of-month (1-based) within each month.
        ym = data.index.to_period("M")
        tdom = pd.Series(1.0, index=data.index).groupby(ym).cumsum()
        out["tdom"] = tdom

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalAccelParams,
    ) -> SignalFrame:
        n = len(data)
        df = pd.DataFrame(index=data.index)

        accel = indicators["accel"].to_numpy(dtype=float)
        accel_prev = indicators["accel_prev"].to_numpy(dtype=float)
        trend_ok = indicators["trend_ok"].to_numpy(dtype=float)
        tdom = indicators["tdom"].to_numpy(dtype=float)

        eps = float(params.accel_eps)
        lo = int(params.tdom_lo)
        hi = int(params.tdom_hi)
        hold = max(int(params.hold_bars), 1)

        # NaN comparisons evaluate False, so warmup bars never trigger an entry.
        cross_up = (accel > eps) & (accel_prev <= eps)
        in_window = (tdom >= lo) & (tdom <= hi)
        trend_gate = trend_ok > 0.5
        entry = cross_up & in_window & trend_gate
        entry = np.where(np.isnan(accel) | np.isnan(accel_prev), False, entry)

        # Fixed-bar exit: hold exactly `hold` bars after entry, no early exit,
        # and ignore fresh entry triggers while a position is open.
        signal = np.zeros(n, dtype=int)
        bars_left = 0
        for i in range(n):
            if bars_left > 0:
                signal[i] = 1
                bars_left -= 1
            elif entry[i]:
                signal[i] = 1
                bars_left = hold - 1

        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
