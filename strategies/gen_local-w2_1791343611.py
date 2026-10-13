from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPolarityParams:
    alternation_window: int = 12
    minimum_alternation: float = 0.65
    minimum_gap: float = 0.001
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapPolarityParams]):
    strategy_id = "gen_local-w2_1791343611"

    @classmethod
    def params_type(cls) -> type[GapPolarityParams]:
        return GapPolarityParams

    def warmup_bars(self, params: GapPolarityParams) -> int:
        return max(200, params.alternation_window + 4)

    def indicators(self, data: pd.DataFrame, params: GapPolarityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["previous_gap"] = out["gap"].shift(1)
        valid_pair = out["gap"].notna() & out["previous_gap"].notna()
        out["polarity_flip"] = (out["gap"] * out["previous_gap"] < 0).astype(float).where(valid_pair)
        out["prior_alternation"] = out["polarity_flip"].rolling(
            params.alternation_window, min_periods=params.alternation_window
        ).mean().shift(2)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["session_return"] = (data["close"] / data["open"].where(lambda s: s > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["previous_session_return"] = out["session_return"].shift(1)
        out["entry_ready"] = (
            (data["close"] > out["ma200"])
            & (out["prior_alternation"] >= params.minimum_alternation)
            & (out["gap"] >= params.minimum_gap)
            & (out["previous_gap"] >= params.minimum_gap)
            & (out["session_return"] >= 0.0)
            & (out["previous_session_return"] >= 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPolarityParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
                continue
            if entries[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapPolarityParams()
    index = pd.date_range("2020-01-01", periods=19, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_ready": True}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0] + [1] * 8 + [0] + [1] * 8 + [0]
    assert result.data["size"].gt(0).all()
