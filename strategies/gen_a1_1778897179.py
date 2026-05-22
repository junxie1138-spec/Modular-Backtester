from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietGapParams:
    gap_threshold_pct: float = 0.15
    range_lookback: int = 20
    compression_ratio: float = 0.85
    profit_target_pct: float = 2.0
    time_stop_bars: int = 2
    size_scale: float = 1.0


class GeneratedStrategy(BaseStrategy[QuietGapParams]):
    strategy_id = "gen_a1_1778897179"

    @classmethod
    def params_type(cls) -> type[QuietGapParams]:
        return QuietGapParams

    @staticmethod
    def warmup_bars(params: QuietGapParams) -> int:
        return int(max(params.range_lookback, 2)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: QuietGapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        prior_close = close.shift(1)

        # Primitive 1: gap behavior - overnight gap as percent of prior close.
        out["gap_pct"] = (data["open"] - prior_close) / prior_close.replace(0.0, np.nan) * 100.0

        # Primitive 2: high-low range dynamics - true range vs its rolling median.
        hl = data["high"] - data["low"]
        hc = (data["high"] - prior_close).abs()
        lc = (data["low"] - prior_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out["true_range"] = tr

        lb = int(max(params.range_lookback, 2))
        tr_median = tr.rolling(lb, min_periods=lb).median()
        out["tr_median"] = tr_median
        out["compression"] = tr / tr_median.replace(0.0, np.nan)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuietGapParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        gap = indicators["gap_pct"].to_numpy(dtype=float)
        comp = indicators["compression"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        n = len(close)
        gthr = float(params.gap_threshold_pct)
        cratio = float(params.compression_ratio)
        pt = float(params.profit_target_pct) / 100.0
        tstop = int(max(params.time_stop_bars, 1))
        sscale = float(params.size_scale)

        # Two-primitive AND: gap sign AND compressed intraday range must agree.
        raw_long = np.zeros(n, dtype=bool)
        raw_short = np.zeros(n, dtype=bool)
        for i in range(n):
            g = gap[i]
            c = comp[i]
            if not np.isfinite(g) or not np.isfinite(c):
                continue
            if c < cratio:
                if g > gthr:
                    raw_long[i] = True
                elif g < -gthr:
                    raw_short[i] = True

        position = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        pos = 0
        entry_price = 0.0
        bars_in = 0
        for i in range(n):
            if pos == 0:
                if raw_long[i]:
                    pos = 1
                    entry_price = close[i]
                    bars_in = 0
                elif raw_short[i]:
                    pos = -1
                    entry_price = close[i]
                    bars_in = 0
            else:
                bars_in += 1
                if entry_price > 0.0 and np.isfinite(close[i]):
                    if pos == 1:
                        pnl = (close[i] - entry_price) / entry_price
                    else:
                        pnl = (entry_price - close[i]) / entry_price
                else:
                    pnl = 0.0
                # Exit: profit-target OR time-stop, whichever fires first.
                if pnl >= pt or bars_in >= tstop:
                    pos = 0
            position[i] = pos

            if pos != 0:
                c = comp[i]
                if np.isfinite(c):
                    conv = 1.0 + (cratio - c) * sscale
                else:
                    conv = 1.0
                size[i] = min(max(conv, 0.5), 1.5)
            else:
                size[i] = 1.0

        df["signal"] = position
        df["size"] = size

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
