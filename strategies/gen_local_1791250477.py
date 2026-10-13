from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    memory: int = 4
    percentile_window: int = 63
    entry_quantile: float = 0.7


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791250477"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.memory + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.memory < 1 or params.percentile_window < 2:
            raise ValueError("Lookbacks must be positive; percentile_window must be at least two.")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must lie strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["width"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["dilation"] = (out["width"].diff() / out["width"].shift(1)).clip(lower=0)
        out["migration"] = out["midpoint"].diff() / out["width"].shift(1)
        out["impulse"] = out["dilation"].shift(1) * out["migration"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["score"] = out["impulse"].rolling(params.memory, min_periods=params.memory).mean()
        out["threshold"] = (
            out["score"].abs().shift(1)
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.entry_quantile)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        score = indicators["score"]
        threshold = indicators["threshold"]
        valid = np.isfinite(score) & np.isfinite(threshold)
        entries = np.select(
            [
                valid & (score > threshold) & (score > 0),
                valid & (score < -threshold) & (score < 0),
            ],
            [1, -1],
            default=0,
        )
        state = 0
        signal = np.zeros(len(data), dtype=int)
        for i, candidate in enumerate(entries):
            if state == 0:
                state = int(candidate)
            elif candidate == -state:
                # Opposite entry evidence exits to flat for this decision bar.
                state = 0
            signal[i] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame(
        {"score": [np.nan, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0], "threshold": 1.0},
        index=index,
    )
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
