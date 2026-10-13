from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    risk_window: int = 4
    rank_window: int = 63
    rank_threshold: float = 0.30
    target_vol: float = 0.12
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791235636"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.risk_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.risk_window < 2 or params.rank_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.rank_threshold < 1.0:
            raise ValueError("rank_threshold must be between zero and one")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0.0:
            raise ValueError("target_vol must be finite and positive")
        if not np.isfinite(params.max_size) or not 0.0 < params.max_size <= 1.0:
            raise ValueError("max_size must be in (0, 1]")

        out = pd.DataFrame(index=data.index)
        out["day_return"] = np.log(data["close"].where(data["close"] > 0) / data["open"].where(data["open"] > 0))
        out["gap_return"] = np.log(data["open"].where(data["open"] > 0) / data["close"].shift(1).where(data["close"].shift(1) > 0))
        out["close_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        w = params.risk_window
        out["day_energy"] = out["day_return"].pow(2).rolling(w, min_periods=w).sum()
        out["gap_energy"] = out["gap_return"].pow(2).rolling(w, min_periods=w).sum()
        total = out["day_energy"] + out["gap_energy"]
        out["day_variance_share"] = out["day_energy"] / total.where(total > 0.0)
        out["share_rank"] = out["day_variance_share"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["day_drift"] = out["day_return"].rolling(w, min_periods=w).sum()
        out["annual_vol"] = out["close_return"].rolling(w, min_periods=w).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        valid = indicators[["share_rank", "day_drift", "annual_vol"]].notna().all(axis=1)
        condition = valid & (indicators["share_rank"] <= params.rank_threshold) & (indicators["day_drift"] > 0.0)
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        # Stay invested until the original eligibility condition flips false.
        df["signal"] = pd.Series(np.where(~condition, 0.0, np.where(confirmed, 1.0, np.nan)), index=data.index).ffill().fillna(0).astype(int)
        decision_size = (params.target_vol / indicators["annual_vol"].clip(lower=1e-8)).clip(lower=1e-6, upper=params.max_size)
        df["size"] = decision_size.shift(1).fillna(params.max_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"share_rank": [0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1], "day_drift": 0.01, "annual_vol": 0.20}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1]
    assert np.allclose(result["size"].iloc[1:], 0.60)
