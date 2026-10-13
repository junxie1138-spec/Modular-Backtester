from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    drift_window: int = 4
    rank_window: int = 63
    vol_window: int = 20
    entry_quantile: float = 0.80
    target_vol: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791288516"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(params.drift_window + 2, params.vol_window + 1) + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if min(params.drift_window, params.rank_window, params.vol_window) < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not (0 < params.entry_quantile < 1 and 0 < params.max_size <= 1):
            raise ValueError("Invalid quantile or exposure cap.")
        if not (np.isfinite(params.target_vol) and params.target_vol > 0):
            raise ValueError("Target volatility must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["latent_slope"] = out["return"].rolling(params.drift_window).mean()
        out["predicted_slope"] = out["latent_slope"].shift(1)
        out["innovation"] = out["return"] - out["predicted_slope"]
        out["volatility"] = out["return"].rolling(params.vol_window).std(ddof=0)
        out["tension"] = (-out["predicted_slope"]).clip(lower=0)
        variance = out["volatility"].pow(2).where(out["volatility"] > 0)
        out["release_score"] = out["tension"] * out["innovation"] / variance
        out["threshold"] = out["release_score"].shift(1).rolling(params.rank_window).quantile(params.entry_quantile)
        out["exposure"] = (params.target_vol / (out["volatility"] * np.sqrt(252))).clip(upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticParams,
    ) -> SignalFrame:
        valid = indicators[["predicted_slope", "innovation", "release_score", "threshold", "exposure"]].notna().all(axis=1)
        entry = (
            valid
            & indicators["predicted_slope"].lt(0)
            & indicators["innovation"].gt(0)
            & indicators["release_score"].gt(indicators["threshold"])
        )
        df = pd.DataFrame(index=data.index)
        # A position persists exactly while the entry condition remains true.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(params.max_size).clip(lower=0.000001, upper=params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = ElasticParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    ind = pd.DataFrame({
        "predicted_slope": [-1.0] * 6,
        "innovation": [1.0] * 6,
        "release_score": [np.nan, 2.0, 3.0, 0.5, 4.0, 0.0],
        "threshold": [1.0] * 6,
        "exposure": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert np.allclose(result["size"], [0.95, 0.2, 0.3, 0.4, 0.5, 0.6])


if __name__ == "__main__":
    _self_check()
