from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FortnightRangeParams:
    seasonal_window: int = 252
    momentum_window: int = 7
    seasonal_threshold: float = 0.04
    momentum_threshold: float = 0.002


def _positions(desired: np.ndarray) -> np.ndarray:
    positions = np.zeros(len(desired), dtype=int)
    state = 0
    for i, direction in enumerate(desired):
        direction = int(direction)
        if state == 0:
            state = direction
        elif direction == -state:
            state = 0
        positions[i] = state
    return positions


class GeneratedStrategy(BaseStrategy[FortnightRangeParams]):
    strategy_id = "gen_local-w2_1791344081"

    @classmethod
    def params_type(cls) -> type[FortnightRangeParams]:
        return FortnightRangeParams

    @staticmethod
    def warmup_bars(params: FortnightRangeParams) -> int:
        return int(max(params.seasonal_window + 1, params.momentum_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FortnightRangeParams) -> pd.DataFrame:
        if params.seasonal_window < 40 or params.momentum_window < 1:
            raise ValueError("Lookbacks require seasonal_window >= 40 and momentum_window >= 1")
        if not np.isfinite(params.seasonal_threshold) or params.seasonal_threshold <= 0:
            raise ValueError("seasonal_threshold must be finite and positive")
        if not np.isfinite(params.momentum_threshold) or params.momentum_threshold < 0:
            raise ValueError("momentum_threshold must be finite and nonnegative")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        span = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["range_fraction"] = (span / close).replace([np.inf, -np.inf], np.nan)
        location = (2.0 * close - data["high"] - data["low"]) / span.where(span > 0)
        out["close_location"] = location.where(span != 0, 0.0).clip(-1.0, 1.0)
        out["signed_range"] = out["range_fraction"] * out["close_location"]
        out["first_half"] = (data.index.day <= 15).astype(int)
        valid = out["range_fraction"].notna() & out["signed_range"].notna()

        for phase in (0, 1):
            mask = out["first_half"].eq(phase)
            signed = out["signed_range"].where(mask, 0.0)
            magnitude = out["range_fraction"].where(mask, 0.0)
            out[f"signed_sum_{phase}"] = signed.rolling(params.seasonal_window, min_periods=params.seasonal_window).sum().shift(1)
            out[f"range_sum_{phase}"] = magnitude.rolling(params.seasonal_window, min_periods=params.seasonal_window).sum().shift(1)
            out[f"observations_{phase}"] = (mask & valid).astype(float).rolling(params.seasonal_window, min_periods=params.seasonal_window).sum().shift(1)
            denominator = out[f"range_sum_{phase}"].where(out[f"range_sum_{phase}"] > 0)
            score = out[f"signed_sum_{phase}"] / denominator
            out[f"seasonal_score_{phase}"] = score.where(out[f"observations_{phase}"] >= 20)

        out["seasonal_score"] = out["seasonal_score_1"].where(out["first_half"].eq(1), out["seasonal_score_0"])
        out["weekly_momentum"] = (close / close.shift(params.momentum_window) - 1.0).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FortnightRangeParams,
    ) -> SignalFrame:
        seasonal = indicators["seasonal_score"]
        momentum = indicators["weekly_momentum"]
        long_entry = seasonal.gt(params.seasonal_threshold) & momentum.gt(params.momentum_threshold)
        short_entry = seasonal.lt(-params.seasonal_threshold) & momentum.lt(-params.momentum_threshold)
        desired = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = _positions(desired)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Neutral observations retain exposure; opposite agreement exits before re-entry.
    desired = np.array([1, 0, -1, -1, 0, 1])
    positions = _positions(desired)
    assert positions.tolist() == [1, 1, 0, -1, -1, 0]
    shifted = pd.Series(positions).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 1, 1, 0, -1, -1]
