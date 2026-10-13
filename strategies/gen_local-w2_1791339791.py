from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    baseline_bars: int = 5
    shock_ratio: float = 1.8
    relaxation_ratio: float = 0.65
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791339791"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        if not 2 <= params.baseline_bars <= 8:
            raise ValueError("baseline_bars must be between 2 and 8")
        if not 15 <= params.holding_bars <= 20:
            raise ValueError("holding_bars must be between 15 and 20")
        if not np.isfinite(params.shock_ratio) or params.shock_ratio <= 1.0:
            raise ValueError("shock_ratio must be finite and greater than one")
        if not np.isfinite(params.relaxation_ratio) or not 0 < params.relaxation_ratio < 1:
            raise ValueError("relaxation_ratio must be between zero and one")
        return int(params.baseline_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["baseline"] = out["range"].shift(1).rolling(params.baseline_bars, min_periods=params.baseline_bars).median()
        out["expansion"] = out["range"].div(out["baseline"].where(out["baseline"] > 0))
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["body_direction"] = np.sign(data["close"] - data["open"])
        out["shock"] = (out["baseline"] > 0) & (out["expansion"] >= params.shock_ratio)
        out["wave_direction"] = out["body_direction"].shift(1)
        prior_range = out["range"].shift(1)
        out["relaxation"] = out["range"].div(prior_range.where(prior_range > 0))
        out["transmission"] = out["midpoint"].diff().mul(out["wave_direction"]).div(prior_range.where(prior_range > 0))
        out["edge_pressure"] = (data["close"] - out["midpoint"]).mul(out["wave_direction"]).div(out["range"].where(out["range"] > 0))
        ready = (
            out["shock"].shift(1, fill_value=False)
            & (out["range"] > 0)
            & (out["relaxation"] <= params.relaxation_ratio)
            & (out["transmission"] >= 0.10)
            & (out["edge_pressure"] >= 0.25)
        )
        out["candidate"] = out["wave_direction"].where(ready, 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def _fixed_positions(candidates: np.ndarray, holding_bars: int) -> np.ndarray:
        positions = np.zeros(len(candidates), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + holding_bars
            positions[i] = position
        return positions

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        df["signal"] = GeneratedStrategy._fixed_positions(candidates, params.holding_bars)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    candidates = np.array([1, -1, 0, 1, -1, 1, 0, -1])
    positions = GeneratedStrategy._fixed_positions(candidates, 3)
    filled = pd.Series(positions).shift(1).fillna(0).astype(int)
    assert filled.tolist() == [0, 1, 1, 1, 0, -1, -1, -1]
    assert GeneratedStrategy.warmup_bars(ShockwaveParams(baseline_bars=8)) == 10
