from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    peak_window: int = 63
    repair_window: int = 8
    volume_window: int = 20
    min_depth: float = 0.015
    entry_threshold: float = 0.35
    full_strength: float = 1.0
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = "gen_local_1790927939"

    @classmethod
    def params_type(cls) -> type[RepairParams]:
        return RepairParams

    @staticmethod
    def warmup_bars(params: RepairParams) -> int:
        return int(max(
            params.peak_window + params.repair_window,
            params.volume_window + params.repair_window,
            params.repair_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        if min(params.peak_window, params.repair_window, params.volume_window) < 1:
            raise ValueError("Lookback windows must be positive")
        if not (0 < params.entry_threshold <= params.full_strength):
            raise ValueError("Require 0 < entry_threshold <= full_strength")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Require 0 < min_size <= max_size <= 1")
        if params.min_depth < 0:
            raise ValueError("min_depth must be nonnegative")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = (1.0 - close / out["peak"]).clip(lower=0)
        out["reference_depth"] = out["drawdown"].shift(params.repair_window)
        out["reference_peak"] = out["peak"].shift(params.repair_window)
        out["stable_peak"] = out["peak"].eq(out["reference_peak"])
        out["return"] = close.pct_change(fill_method=None)
        out["path_length"] = out["return"].abs().rolling(
            params.repair_window, min_periods=params.repair_window
        ).sum()
        out["volume_baseline"] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out["relative_volume"] = volume / out["volume_baseline"].replace(0, np.nan)
        out["participation"] = out["relative_volume"].rolling(
            params.repair_window, min_periods=params.repair_window
        ).mean()
        out["repair"] = out["reference_depth"] - out["drawdown"]
        denominator = out["path_length"] * out["participation"]
        score = out["repair"] / denominator.replace(0, np.nan)
        valid = (
            out["stable_peak"]
            & out[["drawdown", "reference_depth"]].max(axis=1).ge(params.min_depth)
            & np.isfinite(score)
            & out["participation"].gt(0)
        )
        out["eligible"] = valid.fillna(False)
        out["score"] = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RepairParams,
    ) -> SignalFrame:
        score = indicators["score"].fillna(0.0).to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        candidates = np.where(
            eligible & (score >= params.entry_threshold), 1,
            np.where(eligible & (score <= -params.entry_threshold), -1, 0),
        )
        entry_sizes = params.min_size + (params.max_size - params.min_size) * np.clip(
            np.abs(score) / params.full_strength, 0.0, 1.0
        )
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        held_size = params.min_size
        for i, candidate in enumerate(candidates):
            if position == 0 and candidate != 0:
                position = int(candidate)
                held_size = float(entry_sizes[i])
            elif position != 0 and candidate == -position:
                position = 0
                held_size = params.min_size
            signals[i] = position
            sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RepairParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame(
        {"score": [0.5, 0.0, -0.8, -0.8, 0.0, 0.5, 0.5, 0.0], "eligible": True},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
    assert result["size"].iloc[4] > result["size"].iloc[1]
    assert result["size"].iloc[2] == result["size"].iloc[1]
