from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    ma_bars: int = 30
    z_entry: float = 1.0
    gap_min: float = 0.001
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = "gen_local_1791297890"

    @classmethod
    def params_type(cls) -> type[GapAcceptanceParams]:
        return GapAcceptanceParams

    @classmethod
    def warmup_bars(cls, params: GapAcceptanceParams) -> int:
        return params.ma_bars + 2

    def indicators(self, data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma"] = data["close"].rolling(params.ma_bars).mean()
        out["std"] = data["close"].rolling(params.ma_bars).std(ddof=0)
        out["z"] = ((data["close"] - out["ma"]) / out["std"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["prior_high"] = data["high"].shift(1)
        out["gap"] = (data["open"] / out["prior_high"].replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["setup"] = (out["gap"] >= params.gap_min) & (out["z"] >= params.z_entry)
        out["gap_floor"] = out["prior_high"].shift(2)
        out["entry_ready"] = (
            out["setup"].shift(2, fill_value=False)
            & (data["low"].shift(1) > out["gap_floor"])
            & (data["low"] > out["gap_floor"])
            & (out["z"].shift(1) >= params.z_entry / 2.0)
            & (out["z"] >= params.z_entry / 2.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAcceptanceParams,
    ) -> SignalFrame:
        ready = indicators["entry_ready"].to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if i == exit_at:
                exit_at = -1
                continue
            if exit_at >= 0:
                target[i] = 1
            elif ready[i]:
                target[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
