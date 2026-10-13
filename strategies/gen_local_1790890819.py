from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    horizon: int = 18
    congestion_threshold: float = 0.50
    innovation_threshold: float = 0.18
    pressure_threshold: float = 0.05

    def __post_init__(self):
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 10:
            raise ValueError("horizon must be an integer >= 10")
        if not 0.0 < self.congestion_threshold < 1.0:
            raise ValueError("congestion_threshold must be between zero and one")
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0.0:
            raise ValueError("innovation_threshold must be positive and finite")
        if not 0.0 < self.pressure_threshold < 1.0:
            raise ValueError("pressure_threshold must be between zero and one")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790890819"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(2 * params.horizon + 1, params.horizon + 9)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        n = params.horizon
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        out["log_return"] = out["log_close"].diff()
        t = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["sum_log"] = out["log_close"].rolling(n, min_periods=n).sum()
        out["sum_time_log"] = (t * out["log_close"]).rolling(n, min_periods=n).sum()
        # Local linear regression estimates the unobserved daily trend slope.
        out["slope"] = (
            out["sum_time_log"] - (t - (n - 1) / 2.0) * out["sum_log"]
        ) / (n * (n * n - 1) / 12.0)
        out["slope_update"] = out["slope"].diff(5)
        out["noise"] = out["log_return"].rolling(2 * n, min_periods=2 * n).std(ddof=0)
        out["innovation"] = (
            out["slope_update"].rolling(5, min_periods=5).mean()
            / (out["noise"].where(out["noise"] > 1e-12) / np.sqrt(n))
        )
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["intersection"] = (
            np.minimum(data["high"], data["high"].shift(1))
            - np.maximum(data["low"], data["low"].shift(1))
        ).clip(lower=0.0)
        out["union"] = (
            np.maximum(data["high"], data["high"].shift(1))
            - np.minimum(data["low"], data["low"].shift(1))
        )
        out["overlap"] = out["intersection"] / out["union"].where(out["union"] > 0.0)
        out["congestion"] = out["overlap"].rolling(2 * n, min_periods=2 * n).mean()
        out["body"] = data["close"] - data["open"]
        out["body_sum"] = out["body"].rolling(n, min_periods=n).sum()
        out["range_sum"] = out["range"].rolling(n, min_periods=n).sum()
        out["pressure"] = out["body_sum"] / out["range_sum"].where(out["range_sum"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        valid = indicators[["innovation", "congestion", "pressure"]].notna().all(axis=1)
        congested = indicators["congestion"] >= params.congestion_threshold
        directed = indicators["innovation"].where(~congested, -indicators["innovation"])
        long_entry = (
            valid
            & (directed > params.innovation_threshold)
            & (indicators["pressure"] > params.pressure_threshold)
        )
        short_entry = (
            valid
            & (directed < -params.innovation_threshold)
            & (indicators["pressure"] < -params.pressure_threshold)
        )
        entry = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(entry):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposite AND-qualified entry closes first; no same-bar reversal.
                position = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {
            "innovation": [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "congestion": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "pressure": [1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0],
        },
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, 0, 1]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
