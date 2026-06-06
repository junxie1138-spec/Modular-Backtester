from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TurnOfMonthSpringParams:
    range_window: int = 8          # rolling lookback for range compression / close position
    comp_thresh: float = 0.55      # current bar range / rolling-max range below this = coiled spring
    pos_thresh: float = 0.40       # close position inside rolling range below this = depressed
    tom_last_days: int = 4         # last N trading days of a calendar month count as turn-of-month
    tom_first_days: int = 3        # first N trading days of a calendar month count as turn-of-month
    profit_target: float = 0.04    # exit at +4% gain from entry close
    time_stop: int = 10            # exit after at most 10 bars (~2 weeks) held
    base_size: float = 1.0         # per-signal size; portfolio layer scales to equity


class GeneratedStrategy(BaseStrategy[TurnOfMonthSpringParams]):
    strategy_id = "gen_a1_1778887113"

    @classmethod
    def params_type(cls):
        return TurnOfMonthSpringParams

    @staticmethod
    def warmup_bars(params: TurnOfMonthSpringParams) -> int:
        return int(max(2, params.range_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: TurnOfMonthSpringParams) -> pd.DataFrame:
        idx = data.index
        high = data["high"]
        low = data["low"]
        close = data["close"]

        w = int(max(2, params.range_window))

        # --- spring tension: how compressed is the current bar's range vs recent travel ---
        rng = (high - low)
        rng_max = rng.rolling(w, min_periods=1).max()
        comp = rng / rng_max.replace(0.0, np.nan)

        # --- displacement: where the close sits inside the recent high-low envelope ---
        hh = high.rolling(w, min_periods=1).max()
        ll = low.rolling(w, min_periods=1).min()
        span = (hh - ll).replace(0.0, np.nan)
        close_pos = (close - ll) / span

        # --- calendar: trading-day-of-month, no warmup cost (derived from the index) ---
        ym = pd.Series(idx.year * 100 + idx.month, index=idx)
        tdom = ym.groupby(ym).cumcount()                       # 0-based trading day within month
        grp_size = ym.map(ym.value_counts())                   # trading days in that month
        tdom_from_end = grp_size.astype(float) - tdom.astype(float) - 1.0

        last_days = int(max(1, params.tom_last_days))
        first_days = int(max(1, params.tom_first_days))
        in_tom = ((tdom_from_end < last_days) | (tdom < first_days)).astype(float)

        out = pd.DataFrame(index=idx)
        out["comp"] = comp.fillna(1.0)            # NaN during warmup -> treat as uncompressed
        out["close_pos"] = close_pos.fillna(0.5)  # NaN -> treat as mid-range (neutral)
        out["in_tom"] = in_tom.fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TurnOfMonthSpringParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        comp = indicators["comp"].to_numpy(dtype=float)
        cpos = indicators["close_pos"].to_numpy(dtype=float)
        in_tom = indicators["in_tom"].to_numpy(dtype=float)
        n = len(close)

        comp_thresh = float(params.comp_thresh)
        pos_thresh = float(params.pos_thresh)
        profit_target = float(params.profit_target)
        time_stop = int(max(1, params.time_stop))

        raw = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                # profit-target + time-stop exit, whichever fires first
                if gain >= profit_target or bars_held >= time_stop:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                coiled = comp[i] < comp_thresh
                depressed = cpos[i] < pos_thresh
                seasonal = in_tom[i] > 0.5
                if coiled and depressed and seasonal:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(max(1e-6, params.base_size))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
