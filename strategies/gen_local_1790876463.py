from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory_window: int = 40
    volume_window: int = 20
    volume_ratio: float = 1.2
    reversal_strength: float = 0.2

    def __post_init__(self):
        if self.memory_window < 4 or self.volume_window < 2:
            raise ValueError("Lookbacks must be at least 4 and 2.")
        if not np.isfinite(self.volume_ratio) or self.volume_ratio <= 0:
            raise ValueError("volume_ratio must be positive and finite.")
        if not np.isfinite(self.reversal_strength) or not 0 < self.reversal_strength <= 1:
            raise ValueError("reversal_strength must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790876463"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.volume_window + params.memory_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["direction"] = np.sign(out["return"])
        out["volume_baseline"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out["relative_volume"] = (
            data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["volume_event"] = (
            out["relative_volume"].ge(params.volume_ratio)
            & out["direction"].notna()
            & out["direction"].ne(0)
        ).astype(float)
        out["lag_direction"] = out["direction"].shift(1)
        out["eligible_pair"] = (
            out["volume_event"].shift(1).eq(1)
            & out["direction"].notna()
            & out["direction"].ne(0)
            & out["lag_direction"].notna()
            & out["lag_direction"].ne(0)
        ).astype(float)
        out["signed_pair"] = (
            out["direction"] * out["lag_direction"]
        ).where(out["eligible_pair"].eq(1), 0.0)
        out["pair_count"] = out["eligible_pair"].rolling(
            params.memory_window, min_periods=params.memory_window
        ).sum()
        out["pair_sum"] = out["signed_pair"].rolling(
            params.memory_window, min_periods=params.memory_window
        ).sum()
        out["conditional_memory"] = (
            out["pair_sum"] / out["pair_count"].where(out["pair_count"] >= 4)
        ).shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        volume_decline = (
            indicators["direction"].lt(0)
            & indicators["relative_volume"].ge(params.volume_ratio)
        )
        reversal_memory = indicators["conditional_memory"].le(
            -params.reversal_strength
        )
        df = pd.DataFrame(index=data.index)
        # A held long exits as soon as the joint entry condition flips false.
        df["signal"] = (volume_decline & reversal_memory).fillna(False).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = pd.DataFrame(
        {
            "direction": [-1, -1, -1, -1, 1, -1, -1],
            "relative_volume": [2, 2, 2, 1, 2, 2, 2],
            "conditional_memory": [-0.5, -0.5, 0.5, -0.5, -0.5, -0.5, np.nan],
        },
        index=index,
    )
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 1]
    assert result["size"].gt(0).all()
    flat = strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params
    ).data
    assert flat["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
