from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectionReleaseParams:
    rank_window: int = 60
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RejectionReleaseParams]):
    strategy_id = "gen_local_1790996939"

    @classmethod
    def params_type(cls) -> type[RejectionReleaseParams]:
        return RejectionReleaseParams

    def warmup_bars(self, params: RejectionReleaseParams) -> int:
        return int(max(20, max(2, int(params.rank_window)) + 32))

    def indicators(self, data: pd.DataFrame, params: RejectionReleaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["body_top"] = data[["open", "close"]].max(axis=1)
        out["upper_wick"] = (data["high"] - out["body_top"]).clip(lower=0.0)
        out["normalized_rejection"] = out["upper_wick"] / out["atr"].where(out["atr"] > 0.0)
        out["rejection_burden"] = out["normalized_rejection"].rolling(5, min_periods=5).mean()
        window = max(2, int(params.rank_window))
        out["rejection_percentile"] = out["rejection_burden"].rolling(window, min_periods=window).rank(pct=True)
        out["earlier_rejection_peak"] = out["rejection_percentile"].shift(5).rolling(10, min_periods=10).max()
        out["breakout_level"] = data["high"].shift(1).rolling(20, min_periods=20).max()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["earlier_rejection_peak"] >= 0.80)
            & (out["rejection_percentile"] <= 0.25)
            & (data["close"] > out["breakout_level"])
            & (out["atr"] > 0.0)
            & np.isfinite(data["close"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RejectionReleaseParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        age = 0
        fixed_stop = np.nan
        stop_multiple = max(0.01, float(params.stop_atr))

        for i in range(len(data)):
            if active:
                if pending:
                    # The entry decided yesterday fills at today's open.
                    # Freeze both the entry reference and yesterday's ATR.
                    fixed_stop = opens[i] - stop_multiple * atr[i - 1]
                    pending = False
                    age = 0
                age += 1
                stop_hit = np.isfinite(closes[i]) and closes[i] < fixed_stop
                if stop_hit or age >= 18:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                pending = True
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
