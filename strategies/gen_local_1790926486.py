from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    memory: int = 10
    threshold: float = 0.65


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local_1790926486"

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return 3 * max(2, int(params.memory)) + 5

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        memory = max(2, int(params.memory))
        window = 3 * memory
        alpha = 2.0 / (memory + 1.0)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["latent_slope"] = out["return"].ewm(
            span=memory, adjust=False, min_periods=memory
        ).mean()
        out["slope_innovation"] = out["latent_slope"].diff()
        out["return_noise"] = out["return"].rolling(memory, min_periods=memory).std()
        out["innovation_score"] = out["slope_innovation"].div(
            (alpha * out["return_noise"]).replace(0.0, np.nan)
        )
        out["five_session_return"] = out["return"].rolling(5, min_periods=5).sum()
        out["daily_variance"] = out["return"].rolling(window, min_periods=window).var()
        out["five_session_variance"] = out["five_session_return"].rolling(
            window, min_periods=window
        ).var()
        out["variance_ratio"] = out["five_session_variance"].div(
            (5.0 * out["daily_variance"]).replace(0.0, np.nan)
        )
        out["regime_score"] = out["innovation_score"] * np.where(
            out["variance_ratio"] > 1.0, 1.0, -1.0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _reversal_positions(entries: np.ndarray) -> np.ndarray:
        positions = np.zeros(len(entries), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        return positions

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        threshold = max(0.01, float(params.threshold))
        score = indicators["regime_score"]
        valid = indicators[["innovation_score", "variance_ratio"]].notna().all(axis=1)
        entries = np.select(
            [valid & (score >= threshold), valid & (score <= -threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._reversal_positions(entries)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    entries = np.array([1, 0, 1, -1, -1, 0, 1, 1], dtype=int)
    expected = np.array([1, 1, 1, 0, -1, -1, 0, 1], dtype=int)
    assert np.array_equal(GeneratedStrategy._reversal_positions(entries), expected)
