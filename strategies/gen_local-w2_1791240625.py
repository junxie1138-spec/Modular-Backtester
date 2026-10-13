from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownCentroidParams:
    peak_window: int = 20
    centroid_window: int = 3
    minimum_depth: float = 0.01
    excess_depth: float = 0.003


class GeneratedStrategy(BaseStrategy[DrawdownCentroidParams]):
    strategy_id = "gen_local-w2_1791240625"

    @classmethod
    def params_type(cls) -> type[DrawdownCentroidParams]:
        return DrawdownCentroidParams

    def warmup_bars(self, params: DrawdownCentroidParams) -> int:
        return int(params.peak_window + params.centroid_window + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownCentroidParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.centroid_window < 2:
            raise ValueError("Rolling windows must be at least two bars.")
        if not (np.isfinite(params.minimum_depth) and np.isfinite(params.excess_depth)):
            raise ValueError("Depth thresholds must be finite.")
        if params.minimum_depth <= 0 or params.excess_depth <= 0:
            raise ValueError("Depth thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["centroid"] = out["depth"].rolling(
            params.centroid_window, min_periods=params.centroid_window
        ).mean()
        out["excess"] = out["depth"] - out["centroid"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            out[["depth", "centroid", "excess"]].notna().all(axis=1)
            & out["depth"].ge(params.minimum_depth)
            & out["excess"].ge(params.excess_depth)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownCentroidParams,
    ) -> SignalFrame:
        eligible = indicators["eligible"].reindex(data.index).fillna(False).astype(bool)
        # Two consecutive eligible closes enter; the first ineligible close exits.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"eligible": [True, True, False, True, True, True, False]}, index=index
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, DrawdownCentroidParams()
    )
    assert result.data["signal"].tolist() == [0, 0, 1, 0, 0, 1, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
