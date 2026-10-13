from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    fractional_order: float = 0.4
    congestion_threshold: float = 0.5
    evidence_step: float = 0.08


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791265177"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 <= params.congestion_threshold <= 1.0:
            raise ValueError("congestion_threshold must be between zero and one")
        if not 0.0 < params.evidence_step <= 1.0:
            raise ValueError("evidence_step must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_open"] = np.log(data["open"].where(data["open"] > 0))
        out["return"] = out["log_close"].diff()
        out["gap"] = out["log_open"] - out["log_close"].shift(1)
        out["intraday"] = out["log_close"] - out["log_open"]
        valid_pair = out[["gap", "intraday"]].notna().all(axis=1)
        out["opposition"] = (out["gap"] * out["intraday"] < 0).astype(float).where(valid_pair)
        out["congestion"] = out["opposition"].rolling(6, min_periods=6).mean()
        weights = np.ones(8, dtype=float)
        for k in range(1, 8):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional = out["log_close"] * weights[0]
        for k in range(1, 8):
            fractional = fractional + weights[k] * out["log_close"].shift(k)
        # Anchor the truncated kernel to remove its residual constant-price response.
        out["fractional_close"] = fractional - weights.sum() * out["log_close"].shift(7)
        out["fractional_level"] = out["fractional_close"].rolling(3, min_periods=3).mean()
        out["fractional_acceleration"] = out["fractional_close"].diff().rolling(2, min_periods=2).mean()
        out["scale"] = out["return"].abs().rolling(6, min_periods=6).mean().clip(lower=1e-12)
        out["congested"] = (out["congestion"] > params.congestion_threshold).astype(float).where(out["congestion"].notna())
        out["direction"] = out["fractional_level"].where(out["congested"] == 0, out["fractional_acceleration"])
        out["normalized_direction"] = out["direction"] / out["scale"]
        ready = out[["fractional_level", "fractional_acceleration", "congestion", "scale"]].notna().all(axis=1)
        out["vote"] = np.sign(out["normalized_direction"]).where(ready)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        evidence = 0.0
        invested = False
        signals = np.zeros(len(data), dtype=int)
        evidence_path = np.zeros(len(data), dtype=float)
        for i, vote in enumerate(indicators["vote"].to_numpy(dtype=float)):
            if np.isfinite(vote):
                evidence = float(np.clip(evidence + params.evidence_step * vote, -1.0, 1.0))
                # The entry predicate is latched until accumulated evidence reverses.
                if evidence >= 0.5:
                    invested = True
                elif evidence <= -0.5:
                    invested = False
            evidence_path[i] = evidence
            signals[i] = int(invested)
        df = pd.DataFrame(index=data.index)
        df["evidence"] = evidence_path
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    close = np.exp(np.concatenate([np.linspace(4.5, 5.1, 60), np.linspace(5.09, 4.4, 60)]))
    opening = close * np.exp(-0.002)
    data = pd.DataFrame({"open": opening, "high": close * 1.01, "low": opening * 0.99, "close": close, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert strategy.warmup_bars(params) == 10
    assert indicators["vote"].iloc[:9].isna().all()
    assert pd.notna(indicators["vote"].iloc[9])
    assert signals.index.equals(data.index)
    assert set(signals["signal"].unique()) <= {0, 1}
    assert (signals["size"] > 0).all()
    assert signals["signal"].iloc[50] == 1
    assert signals["signal"].iloc[-1] == 0
    expected = np.zeros(len(data), dtype=int)
    invested = False
    for i, evidence in enumerate(signals["evidence"]):
        if evidence >= 0.5:
            invested = True
        elif evidence <= -0.5:
            invested = False
        expected[i] = int(invested)
    shifted = np.concatenate([np.zeros(1, dtype=int), expected[:-1]])
    assert np.array_equal(signals["signal"].to_numpy(), shifted)
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_signals = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert (flat_signals["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
