from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    tension_threshold: float = 2.5
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791323094"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 41

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(20, min_periods=20).mean()
        out["peak"] = data["close"].rolling(40, min_periods=40).max()
        out["tension"] = (out["peak"] - data["close"]) / out["atr"].where(out["atr"] > 0)
        out["release"] = out["tension"].shift(1) - out["tension"]
        out["entry"] = (
            (out["tension"].shift(1) >= params.tension_threshold)
            & (out["tension"] >= 1.0)
            & (out["release"] >= 0.5)
            & (out["atr"] < out["atr"].shift(1))
            & (out["peak"] == out["peak"].shift(1))
            & (data["low"] > data["low"].shift(1))
            & (data["close"] > out["previous_close"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not np.isfinite(params.tension_threshold) or params.tension_threshold <= 0:
            raise ValueError("tension_threshold must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")

        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        holding = False
        stop = np.nan
        held_bars = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    holding = True
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    held_bars = 0

            if holding:
                held_bars += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held_bars >= 8:
                    holding = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
