from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    lookback: int = 8
    smoothing: int = 2
    rotation_threshold: float = 0.18
    minimum_mass_share: float = 0.15


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local-w2_1791215497"

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    @staticmethod
    def warmup_bars(params: PressureParams) -> int:
        if params.lookback < 2 or params.smoothing < 1:
            raise ValueError("Lookback must be at least two; smoothing must be positive.")
        warmup = params.lookback + params.smoothing
        if warmup > 10:
            raise ValueError("Combined return and smoothing warmup must not exceed ten bars.")
        if not 0.0 < params.rotation_threshold < 1.0:
            raise ValueError("Rotation threshold must lie strictly between zero and one.")
        if not 0.0 < params.minimum_mass_share < 0.5:
            raise ValueError("Minimum mass share must lie strictly between zero and one half.")
        return int(warmup)

    @staticmethod
    def indicators(data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        n = params.lookback
        result = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        result["return"] = returns
        result["positive_mass"] = returns.clip(lower=0.0)
        result["negative_mass"] = (-returns).clip(lower=0.0)
        for direction in ("positive", "negative"):
            mass = result[f"{direction}_mass"]
            total = mass.rolling(n, min_periods=n).sum()
            weighted = pd.concat(
                [mass.shift(lag) * (n - 1 - lag) for lag in range(n)],
                axis=1,
            ).sum(axis=1, min_count=n)
            result[f"{direction}_total"] = total
            result[f"{direction}_weighted"] = weighted
            result[f"{direction}_centroid"] = weighted.div(
                ((n - 1) * total).where(total > 0.0)
            )
        positive = result["positive_total"]
        negative = result["negative_total"]
        total_mass = positive + negative
        result["minority_share"] = pd.concat([positive, negative], axis=1).min(
            axis=1, skipna=False
        ).div(total_mass.where(total_mass > 0.0))
        result["rotation"] = result["positive_centroid"] - result["negative_centroid"]
        result["smoothed_rotation"] = result["rotation"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).mean()
        result["smoothed_minority_share"] = result["minority_share"].rolling(
            params.smoothing, min_periods=params.smoothing
        ).min()
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PressureParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        rotation = indicators["smoothed_rotation"]
        share = indicators["smoothed_minority_share"]
        ready = rotation.notna() & share.notna() & share.ge(params.minimum_mass_share)
        long_entry = ready & rotation.gt(params.rotation_threshold)
        short_entry = ready & rotation.lt(-params.rotation_threshold)
        df = pd.DataFrame(index=data.index)
        # A held direction exits to zero when its entry predicate flips false.
        df["signal"] = np.select([long_entry, short_entry], [1, -1], default=0)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = PressureParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": np.ones(7)}, index=index)
    features = pd.DataFrame(
        {
            "smoothed_rotation": [np.nan, 0.3, 0.4, 0.0, -0.3, -0.4, 0.0],
            "smoothed_minority_share": [0.3] * 7,
        },
        index=index,
    )
    signals = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert signals["size"].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) == 10
