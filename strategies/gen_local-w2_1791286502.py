from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 18
    area_threshold: float = 0.02

    def __post_init__(self):
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 3:
            raise ValueError("window must be an integer >= 3")
        if not np.isfinite(self.area_threshold) or not 0.0 <= self.area_threshold < 0.5:
            raise ValueError("area_threshold must be finite and in [0, 0.5)")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791286502"

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return 3 * params.window - 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.window
        out = pd.DataFrame(index=data.index)
        out["channel_low"] = data["low"].rolling(n, min_periods=n).min()
        out["channel_high"] = data["high"].rolling(n, min_periods=n).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        width = out["channel_width"].where(out["channel_width"] > 0.0)
        out["position"] = ((data["close"] - out["channel_low"]) / width).replace(
            [np.inf, -np.inf], np.nan
        ).clip(0.0, 1.0)
        out["start_position"] = out["position"].shift(n - 1)
        out["position_sum"] = out["position"].rolling(n, min_periods=n).sum()
        # Antisymmetric level-two signature of (normalized time, position).
        # Positive area means the path lies below its endpoint chord.
        endpoints = out["start_position"] + out["position"]
        out["path_integral"] = (out["position_sum"] - 0.5 * endpoints) / (n - 1)
        out["signature_area"] = 0.5 * endpoints - out["path_integral"]
        out["area_balance"] = out["signature_area"].rolling(n, min_periods=n).mean()
        out["position_balance"] = out["position"].rolling(n, min_periods=n).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        # Persist while this condition holds; its reversal is the only exit.
        entry = (
            indicators["area_balance"].gt(params.area_threshold)
            & indicators["position_balance"].gt(0.5)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams()
    t = np.arange(120, dtype=float)
    close = 100.0 + 0.03 * t + np.sin(t / 7.0)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.full(len(t), 1000.0)},
        index=pd.date_range("2020-01-01", periods=len(t), freq="B"),
    )
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:90], params)
    pd.testing.assert_frame_equal(full.iloc[:90], prefix)
    assert full["area_balance"].first_valid_index() == data.index[
        GeneratedStrategy.warmup_bars(params) - 1
    ]
    probe = full.iloc[:5].copy()
    probe["area_balance"] = [np.nan, 0.04, 0.04, 0.0, 0.04]
    probe["position_balance"] = 0.75
    result = GeneratedStrategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
