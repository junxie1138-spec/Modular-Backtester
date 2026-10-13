from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    review_bars: int = 18
    max_distance: float = 0.35
    min_coarse_share: float = 0.35


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1791292783"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if params.review_bars < 1:
            raise ValueError("review_bars must be positive")
        if not np.isfinite(params.max_distance) or not 0 <= params.max_distance <= 2:
            raise ValueError("max_distance must be between zero and two")
        if not np.isfinite(params.min_coarse_share) or not 0 <= params.min_coarse_share <= 1:
            raise ValueError("min_coarse_share must be between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        r0 = out["return"]
        out["lag1"] = r0.shift(1)
        out["lag2"] = r0.shift(2)
        out["lag3"] = r0.shift(3)
        r1, r2, r3 = out["lag1"], out["lag2"], out["lag3"]
        out["drift"] = r0 + r1 + r2 + r3
        out["fine_energy"] = ((r0 - r1) ** 2 + (r2 - r3) ** 2) / 2.0
        out["medium_energy"] = (r0 + r1 - r2 - r3) ** 2 / 4.0
        out["coarse_energy"] = out["drift"] ** 2 / 4.0
        out["total_energy"] = out[["fine_energy", "medium_energy", "coarse_energy"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        for scale in ("fine", "medium", "coarse"):
            out[scale + "_share"] = out[scale + "_energy"] / denominator
            out["prior_" + scale + "_share"] = out[scale + "_share"].shift(4)
        out["prior_drift"] = out["drift"].shift(4)
        out["distance"] = sum(
            (out[scale + "_share"] - out["prior_" + scale + "_share"]) ** 2
            for scale in ("fine", "medium", "coarse")
        )
        out["valid"] = np.isfinite(out[["distance", "drift", "prior_drift", "coarse_share", "prior_coarse_share"]]).all(axis=1)
        out["entry_condition"] = (
            out["valid"]
            & out["distance"].le(params.max_distance)
            & out["drift"].gt(0)
            & out["prior_drift"].gt(0)
            & out["coarse_share"].ge(params.min_coarse_share)
            & out["prior_coarse_share"].ge(params.min_coarse_share)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        if params.review_bars < 1:
            raise ValueError("review_bars must be positive")
        first_review = GeneratedStrategy.warmup_bars(params) - 1
        positions = np.arange(len(data))
        review = (positions >= first_review) & ((positions - first_review) % params.review_bars == 0)
        # Review cadence changes the entry condition; elapsed time alone never exits.
        condition = indicators["entry_condition"].fillna(False).astype(bool)
        regime = condition.astype(float).where(review).ffill().fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = regime.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShapeParams()
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    changes = np.concatenate([np.full(10, 0.01), np.full(40, -0.01)])
    close = 100.0 * np.exp(np.cumsum(changes))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert GeneratedStrategy.warmup_bars(params) == 9
    assert result["signal"].iloc[:9].eq(0).all()
    assert result["signal"].iloc[9:27].eq(1).all()
    assert result["signal"].iloc[27:].eq(0).all()
    prefix = data.iloc[:30]
    prefix_result = GeneratedStrategy.generate_signals(prefix, GeneratedStrategy.indicators(prefix, params), None, params).data
    assert prefix_result.equals(result.iloc[:30])
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
