from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalElasticDrawdownParams:
    """Parameters for the early-week elastic-drawdown rebound strategy."""

    lookback: int = 10          # rolling window for the drawdown reference high
    dd_shallow: float = 0.02    # minimum drawdown depth to consider (elastic floor)
    dd_deep: float = 0.08       # maximum drawdown depth allowed (plastic ceiling)
    hold_bars: int = 16         # fixed-bar holding horizon (~3.2 weeks)
    max_dow: int = 1            # gate: entry only when weekday index <= this (0=Mon)


class GeneratedStrategy(BaseStrategy[SeasonalElasticDrawdownParams]):
    """Long elastic-zone drawdowns occurring on early-week sessions.

    A drawdown is measured against a short rolling high. Only 'elastic'
    drawdowns - deep enough to matter but not so deep they are 'plastic'
    (regime-breaking) - are tradable, and only when the bar falls on an
    early-week weekday. Each position is closed exactly ``hold_bars`` bars
    after entry (pure fixed-bar exit, no signal-based exit).
    """

    strategy_id = "gen_a2_1779153919"

    @classmethod
    def params_type(cls) -> type[SeasonalElasticDrawdownParams]:
        return SeasonalElasticDrawdownParams

    @staticmethod
    def warmup_bars(params: SeasonalElasticDrawdownParams) -> int:
        # Only lookback needed; rolling max produces NaN for the first
        # (lookback - 1) bars. Clamp to a sane minimum.
        return int(max(2, params.lookback))

    @staticmethod
    def indicators(
        data: pd.DataFrame, params: SeasonalElasticDrawdownParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        win = int(max(2, params.lookback))

        roll_max = close.rolling(win, min_periods=win).max()
        drawdown = close / roll_max - 1.0  # <= 0; NaN during warmup

        dow = pd.Series(
            np.asarray(data.index.dayofweek, dtype="float64"),
            index=data.index,
        )

        out = pd.DataFrame(index=data.index)
        out["drawdown"] = drawdown
        out["dow"] = dow
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalElasticDrawdownParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        dd = indicators["drawdown"]
        dow = indicators["dow"]

        shallow = abs(float(params.dd_shallow))
        deep = abs(float(params.dd_deep))
        max_dow = int(params.max_dow)
        hold = int(max(1, params.hold_bars))

        # Elastic band: drawdown deep enough to be a dip but not 'plastic'.
        # NaN comparisons evaluate False; fillna keeps it explicit.
        entry = (
            (dd <= -shallow)
            & (dd >= -deep)
            & (dow <= float(max_dow))
        )
        entry_arr = entry.fillna(False).to_numpy()

        n = len(df)
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        held = 0
        for i in range(n):
            if in_pos:
                raw[i] = 1
                held += 1
                if held >= hold:
                    in_pos = False
                    held = 0
            elif entry_arr[i]:
                in_pos = True
                held = 1
                raw[i] = 1

        signal = pd.Series(raw, index=df.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
