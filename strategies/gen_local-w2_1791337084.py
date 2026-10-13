from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapVarianceParams:
    shock_ratio: float = 2.5
    coherence_floor: float = 0.85
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapVarianceParams]):
    strategy_id = "gen_local-w2_1791337084"

    @classmethod
    def params_type(cls) -> type[GapVarianceParams]:
        return GapVarianceParams

    def warmup_bars(self, params: GapVarianceParams) -> int:
        # One prior close, two recent gaps, and six earlier gaps.
        return 9

    def indicators(self, data: pd.DataFrame, params: GapVarianceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1))
        older = out["gap"].shift(2)
        out["baseline_rms"] = older.pow(2).rolling(6, min_periods=6).mean().pow(0.5)
        out["baseline_mean"] = older.rolling(6, min_periods=6).mean()
        baseline = out["baseline_rms"].replace(0.0, np.nan)
        out["baseline_bias"] = out["baseline_mean"].abs() / baseline
        out["recent_rms"] = out["gap"].pow(2).rolling(2, min_periods=2).mean().pow(0.5)
        out["variance_jump"] = out["recent_rms"] / baseline
        out["recent_gap"] = out["gap"].rolling(2, min_periods=2).sum()
        out["gap_mass"] = out["gap"].abs().rolling(2, min_periods=2).sum()
        out["coherence"] = out["recent_gap"].abs() / out["gap_mass"].replace(0.0, np.nan)
        out["direction"] = np.sign(out["recent_gap"])
        out["displacement"] = np.log(closing / closing.shift(2))
        out["retention"] = out["direction"] * out["displacement"] / out["recent_gap"].abs().replace(0.0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["baseline_bias", "variance_jump", "coherence", "retention", "direction"]].notna().all(axis=1)
        trigger = (
            ready
            & (out["baseline_bias"] <= 0.5)
            & (out["variance_jump"] >= params.shock_ratio)
            & (out["coherence"] >= params.coherence_floor)
            & (out["retention"] >= 0.5)
        )
        out["candidate"] = out["direction"].where(trigger, 0).fillna(0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapVarianceParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        for i, candidate in enumerate(candidates):
            if position and i == exit_at:
                position = 0
            elif position == 0 and candidate:
                position = int(candidate)
                exit_at = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = GapVarianceParams()
    index = pd.date_range("2020-01-01", periods=24)
    for direction in (1, -1):
        gaps = np.array([0.0] + [0.001, -0.001] * 3 + [0.006, 0.006] + [0.0] * 15) * direction
        prices = 100.0 * np.exp(np.cumsum(gaps))
        data = pd.DataFrame(
            {"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices, "volume": 1000.0},
            index=index,
        )
        derived = strategy.indicators(data, params)
        assert strategy.warmup_bars(params) <= 10
        assert derived["candidate"].iloc[:8].eq(0).all()
        assert derived["candidate"].iloc[8] == direction
        # Opposite entries during a hold and entries on its expiry cannot alter the exit.
        derived.loc[index[10], "candidate"] = -direction
        derived.loc[index[16], "candidate"] = -direction
        signals = strategy.generate_signals(data, derived, None, params).data
        assert signals["signal"].iloc[:9].eq(0).all()
        assert signals["signal"].iloc[9:17].eq(direction).all()
        assert signals["signal"].iloc[17] == 0
        assert signals["size"].gt(0).all()
        flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
        assert strategy.indicators(flat, params)["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
