from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalityParams:
    acceleration_threshold: float = 0.5
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SeasonalityParams]):
    strategy_id = "gen_local-w2_1791271324"

    @classmethod
    def params_type(cls) -> type[SeasonalityParams]:
        return SeasonalityParams

    def warmup_bars(self, params: SeasonalityParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: SeasonalityParams) -> pd.DataFrame:
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold < 0:
            raise ValueError("acceleration_threshold must be finite and nonnegative")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["daily_return"] = close.pct_change(fill_method=None)
        out["roc5"] = close.pct_change(5, fill_method=None)
        out["acceleration"] = out["roc5"].diff(5)
        out["noise"] = out["daily_return"].rolling(20, min_periods=20).std(ddof=0)
        denominator = (out["noise"] * np.sqrt(5.0)).replace(0.0, np.nan)
        out["acceleration_score"] = out["acceleration"] / denominator
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        day = data.index.day
        out["calendar_window"] = (day >= 25) | (day <= 3)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            out["calendar_window"]
            & out["acceleration_score"].gt(params.acceleration_threshold)
            & out["roc5"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalityParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        frozen_atr = np.nan
        stop_price = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                    entry_bar = i
                    active = True
                else:
                    active = False

            if active:
                if (
                    not np.isfinite(closes[i])
                    or closes[i] < stop_price
                    or i - entry_bar + 1 >= 8
                ):
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                frozen_atr = atr[i]
                pending_entry = True
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
