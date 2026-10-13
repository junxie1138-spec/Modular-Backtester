from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    volatility_window: int = 20
    repair_window: int = 20
    volatility_ratio_limit: float = 0.8
    repair_threshold: float = 0.2


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1791328362"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return int(max(params.volatility_window, params.repair_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap_return"] = data["open"].div(out["previous_close"].replace(0.0, np.nan)).sub(1.0)
        out["intraday_return"] = data["close"].div(data["open"].replace(0.0, np.nan)).sub(1.0)
        out["gap_volatility"] = out["gap_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["intraday_volatility"] = out["intraday_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["volatility_ratio"] = out["gap_volatility"].div(out["intraday_volatility"].replace(0.0, np.nan))
        out["downside_gap"] = out["gap_return"].clip(upper=0.0).mul(-1.0)
        out["repair_direction"] = pd.Series(np.where(data["close"] >= out["previous_close"], 1.0, -1.0), index=data.index).where(out["previous_close"].notna())
        out["signed_repair"] = out["downside_gap"].mul(out["repair_direction"])
        out["downside_gap_budget"] = out["downside_gap"].rolling(params.repair_window, min_periods=params.repair_window).sum()
        out["repair_budget"] = out["signed_repair"].rolling(params.repair_window, min_periods=params.repair_window).sum()
        out["repair_balance"] = out["repair_budget"].div(out["downside_gap_budget"].replace(0.0, np.nan))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRepairParams) -> SignalFrame:
        compressed = indicators["volatility_ratio"].le(params.volatility_ratio_limit)
        repaired = indicators["repair_balance"].gt(params.repair_threshold)
        ready = indicators[["volatility_ratio", "repair_balance"]].notna().all(axis=1)
        df = pd.DataFrame(index=data.index)
        df["signal"] = (ready & compressed & repaired).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
