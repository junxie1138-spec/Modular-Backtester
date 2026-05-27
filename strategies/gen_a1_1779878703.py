from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_window: int = 63
    intra_window: int = 63
    gap_low_pct: float = 0.20
    intra_high_pct: float = 0.75
    hysteresis_band: float = 0.10
    confirm_bars: int = 2
    atr_period: int = 14
    k_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779878703"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # Rolling rank windows + ATR period + 1 (pct_change / shift introduces a leading NaN).
        return int(max(params.gap_window, params.intra_window, params.atr_period)) + 5

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        # Overnight gap return: prev close -> today open.
        gap_ret = (open_ - prev_close) / prev_close
        # Intraday return: today open -> today close.
        intra_ret = (close - open_) / open_

        gw = int(params.gap_window)
        iw = int(params.intra_window)

        gap_rank = gap_ret.rolling(gw, min_periods=gw).rank(pct=True)
        intra_rank = intra_ret.rolling(iw, min_periods=iw).rank(pct=True)

        # ATR for the fixed volatility stop.
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_period), min_periods=int(params.atr_period)).mean()

        out = pd.DataFrame(index=data.index)
        out["gap_rank"] = gap_rank
        out["intra_rank"] = intra_rank
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        gap_rank = indicators["gap_rank"].to_numpy(dtype=float)
        intra_rank = indicators["intra_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        gap_strict = float(params.gap_low_pct)
        gap_loose = float(params.gap_low_pct) + float(params.hysteresis_band)
        intra_strict = float(params.intra_high_pct)
        intra_loose = float(params.intra_high_pct) - float(params.hysteresis_band)
        confirm_bars = int(params.confirm_bars)
        k = float(params.k_atr)
        max_hold = int(params.max_hold)

        n = len(close)
        sig = np.zeros(n, dtype=int)

        # Schmitt-trigger decoupling state: enter on strict bands, exit on loose bands.
        state_on = False
        streak = 0

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            gr = gap_rank[i]
            ir = intra_rank[i]

            if np.isnan(gr) or np.isnan(ir):
                state_on = False
                streak = 0
            else:
                if state_on:
                    # stays on unless either rank cleanly leaves its loose band
                    if (gr > gap_loose) or (ir < intra_loose):
                        state_on = False
                else:
                    # turns on only when both ranks satisfy strict criteria simultaneously
                    if (gr < gap_strict) and (ir > intra_strict):
                        state_on = True

                if state_on:
                    streak += 1
                else:
                    streak = 0

            entry_ready = streak >= confirm_bars

            if in_pos:
                bars_held += 1
                # Fixed (entry-anchored, not trailing) ATR vol stop + max-hold cap.
                if close[i] < stop_level or bars_held >= max_hold:
                    in_pos = False
                    bars_held = 0
                    stop_level = 0.0
                    sig[i] = 0
                else:
                    sig[i] = 1
            elif entry_ready:
                atr_i = atr[i]
                if np.isnan(atr_i) or atr_i <= 0.0:
                    sig[i] = 0
                else:
                    in_pos = True
                    bars_held = 0
                    stop_level = close[i] - k * atr_i
                    sig[i] = 1
            else:
                sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
