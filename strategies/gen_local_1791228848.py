from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    box_bars: int = 6
    baseline_bars: int = 42
    evidence_bars: int = 18
    compression_ratio: float = 0.75
    evidence_threshold: float = 0.15
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791228848"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return max(params.box_bars, params.baseline_bars) + params.evidence_bars

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.box_bars, params.baseline_bars, params.evidence_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be between zero and one.")
        if not 0.0 < params.evidence_threshold < 1.0:
            raise ValueError("evidence_threshold must be between zero and one.")
        if not 0.0 < params.minimum_size <= 1.0:
            raise ValueError("minimum_size must be positive and at most one.")
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["fast_range"] = out["daily_range"].rolling(params.box_bars).mean()
        out["baseline_range"] = out["daily_range"].rolling(params.baseline_bars).mean()
        out["box_high"] = data["high"].rolling(params.box_bars).max()
        out["box_low"] = data["low"].rolling(params.box_bars).min()
        out["box_width"] = out["box_high"] - out["box_low"]
        out["compressed"] = (
            (out["baseline_range"] > 0)
            & (out["fast_range"] <= params.compression_ratio * out["baseline_range"])
            & (out["box_width"] <= 0.65 * params.box_bars * out["fast_range"])
        ).fillna(False)
        out["anchor_high"] = out["box_high"].where(out["compressed"]).shift(1).ffill()
        out["anchor_low"] = out["box_low"].where(out["compressed"]).shift(1).ffill()
        out["volume"] = data["volume"].clip(lower=0).fillna(0)
        out["outside_direction"] = (
            (data["close"] > out["anchor_high"]).astype(int)
            - (data["close"] < out["anchor_low"]).astype(int)
        )
        out["signed_volume"] = out["volume"] * out["outside_direction"]
        out["evidence_volume"] = out["signed_volume"].rolling(params.evidence_bars).sum()
        out["total_volume"] = out["volume"].rolling(params.evidence_bars).sum()
        out["score"] = (
            out["evidence_volume"] / out["total_volume"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        # The same condition governs entry and reversal exit; no time exit.
        df["signal"] = (score > params.evidence_threshold).astype(int)
        strength = (
            (score - params.evidence_threshold) / (1.0 - params.evidence_threshold)
        ).clip(0.0, 1.0)
        df["size"] = params.minimum_size + (1.0 - params.minimum_size) * strength
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = CompressionParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=80),
    )
    actual = strategy.indicators(data, params)
    assert actual["score"].notna().all()
    assert actual["score"].eq(0).all()
    probe = pd.DataFrame({"score": [0.0, 0.3, 0.8, 0.0, np.nan]}, index=data.index[:5])
    result = strategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert 0 < result["size"].iloc[2] < result["size"].iloc[3] <= 1
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
