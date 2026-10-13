from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    trend_window: int = 4
    acceptance_window: int = 18
    minimum_retention: float = 0.65


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = "gen_local_1790895092"

    @classmethod
    def params_type(cls) -> type[GapAcceptanceParams]:
        return GapAcceptanceParams

    @staticmethod
    def warmup_bars(params: GapAcceptanceParams) -> int:
        return int(params.fractional_window + max(params.trend_window, params.acceptance_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.trend_window, params.acceptance_window) < 2:
            raise ValueError("lookbacks must be at least two")
        if params.trend_window >= params.acceptance_window:
            raise ValueError("trend_window must be smaller than acceptance_window")
        if not (-1.0 <= params.minimum_retention <= 1.0):
            raise ValueError("minimum_retention must be between minus one and one")

        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = log_close * weights[0]
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + log_close.shift(k) * weights[k]
        out["fractional_close"] = fractional_close
        out["fractional_fast"] = fractional_close.rolling(params.trend_window).mean()
        out["fractional_baseline"] = fractional_close.rolling(params.acceptance_window).mean()

        previous_close = data["close"].shift(1).where(data["close"].shift(1) > 0)
        out["opening_gap"] = data["open"] / previous_close - 1.0
        out["close_return"] = data["close"] / previous_close - 1.0
        out["upside_gap"] = out["opening_gap"].clip(lower=0.0)
        # A retained gap earns its original gap size; rejection can subtract it.
        out["retained_gap"] = out["close_return"].clip(
            lower=-out["upside_gap"], upper=out["upside_gap"]
        ).where(out["opening_gap"].notna())
        out["gap_budget"] = out["upside_gap"].rolling(params.acceptance_window).sum()
        out["retained_budget"] = out["retained_gap"].rolling(params.acceptance_window).sum()
        out["gap_retention"] = out["retained_budget"] / out["gap_budget"].where(out["gap_budget"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAcceptanceParams,
    ) -> SignalFrame:
        condition = (
            (indicators["fractional_fast"] > indicators["fractional_baseline"])
            & (indicators["gap_retention"] >= params.minimum_retention)
        ).fillna(False)
        # Two consecutive qualifying closes enter; the first failed close exits.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapAcceptanceParams()
    close = 100.0 * np.power(1.003, np.arange(100))
    close[-4:] = close[-5] * np.power(0.8, np.arange(1, 5))
    previous = np.r_[close[0] / 1.003, close[:-1]]
    opening = previous * 1.001
    sample = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * 1.001,
        "low": np.minimum(opening, close) * 0.999,
        "close": close,
        "volume": np.full(100, 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=100))
    derived = GeneratedStrategy.indicators(sample, params)
    signals = GeneratedStrategy.generate_signals(sample, derived, None, params).data
    assert signals["signal"].iloc[:50].eq(0).all()
    assert signals["signal"].iloc[-6] == 1
    assert signals["signal"].iloc[-1] == 0
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].gt(0.0).all()
