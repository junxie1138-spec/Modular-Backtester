from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DriftChangeParams:
    slope_bars: int = 18
    percentile_bars: int = 126
    entry_percentile: float = 0.85


class GeneratedStrategy(BaseStrategy[DriftChangeParams]):
    strategy_id = "gen_local-w2_1791258128"

    @classmethod
    def params_type(cls) -> type[DriftChangeParams]:
        return DriftChangeParams

    def warmup_bars(self, params: DriftChangeParams) -> int:
        return int(2 * params.slope_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: DriftChangeParams) -> pd.DataFrame:
        if params.slope_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one")
        n = params.slope_bars
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff().replace([np.inf, -np.inf], np.nan)
        out["slope"] = out["return"].rolling(n, min_periods=n).mean()
        out["previous_slope"] = out["slope"].shift(n)
        out["variance"] = out["return"].rolling(n, min_periods=n).var(ddof=1)
        out["previous_variance"] = out["variance"].shift(n)
        out["innovation"] = out["slope"] - out["previous_slope"]
        out["innovation_noise"] = np.sqrt(
            ((out["variance"] + out["previous_variance"]) / n).clip(lower=0.0)
        )
        out["score"] = (
            out["innovation"] / out["innovation_noise"].clip(lower=np.finfo(float).eps)
        ).replace([np.inf, -np.inf], np.nan)
        out["absolute_score"] = out["score"].abs()
        out["threshold"] = out["absolute_score"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DriftChangeParams,
    ) -> SignalFrame:
        score = indicators["score"]
        threshold = indicators["threshold"]
        slope = indicators["slope"]
        valid = pd.Series(
            np.isfinite(indicators[["score", "threshold", "slope"]].to_numpy()).all(axis=1),
            index=data.index,
        )
        bullish = valid & (score > threshold) & (slope > 0.0)
        bearish = valid & (score < -threshold) & (slope < 0.0)
        events = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {
            "score": [np.nan, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 2.0],
            "threshold": [1.0] * 8,
            "slope": [0.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        },
        index=index,
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, DriftChangeParams()
    ).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
