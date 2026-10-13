from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 5
    stop_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790968272"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(15, 2 * params.roc_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.roc_bars < 1 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("roc_bars and stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        denominator = close.shift(params.roc_bars).where(lambda x: x > 0)
        out["roc"] = close.div(denominator).sub(1).replace([np.inf, -np.inf], np.nan)
        out["prior_roc"] = out["roc"].shift(params.roc_bars)
        out["acceleration"] = out["roc"] - out["prior_roc"]
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["month"] = data.index.year * 12 + data.index.month
        out["opening_week"] = data.index.day <= 7
        out["entry"] = (
            out["opening_week"]
            & out["roc"].gt(0)
            & out["prior_roc"].le(0)
            & out["acceleration"].gt(0)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
            & close.gt(0)
            & np.isfinite(close)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        months = indicators["month"].to_numpy(dtype=int)
        active = False
        entry_decision = -1
        last_entry_month = -1
        highest_close = -np.inf
        stop = -np.inf
        ready_at = GeneratedStrategy.warmup_bars(params) - 1

        for i in range(count):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                # The entry fills on the bar after the entry decision.
                # Only closes observed from that fill bar onward enter the high-water mark.
                if valid:
                    highest_close = max(highest_close, close)
                    stop = max(stop, highest_close - params.stop_atr * atr)
                if not valid or close <= stop or i - entry_decision >= 18:
                    active = False
                else:
                    signals[i] = 1
            elif i >= ready_at and valid and entries[i] and months[i] != last_entry_month:
                active = True
                entry_decision = i
                last_entry_month = months[i]
                highest_close = -np.inf
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": np.ones(count, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
