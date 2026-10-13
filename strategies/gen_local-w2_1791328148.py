from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBudgetParams:
    block_bars: int = 18
    estimation_bars: int = 90
    entry_gap: float = 0.00005


class GeneratedStrategy(BaseStrategy[GapBudgetParams]):
    strategy_id = "gen_local-w2_1791328148"

    @classmethod
    def params_type(cls) -> type[GapBudgetParams]:
        return GapBudgetParams

    def warmup_bars(self, params: GapBudgetParams) -> int:
        return int(2 * params.block_bars + params.estimation_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GapBudgetParams) -> pd.DataFrame:
        if params.block_bars < 2 or params.estimation_bars < 3:
            raise ValueError("Lookbacks must be at least two and three bars respectively")
        if not np.isfinite(params.entry_gap) or params.entry_gap < 0:
            raise ValueError("entry_gap must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        ratio = data["open"].where(data["open"] > 0) / prior_close.where(prior_close > 0)
        out["gap"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        h = params.block_bars
        w = params.estimation_bars
        out["budget"] = out["gap"].rolling(h, min_periods=h).sum()
        out["lagged_budget"] = out["budget"].shift(h)
        out["budget_mean"] = out["budget"].rolling(w, min_periods=w).mean()
        out["lagged_mean"] = out["lagged_budget"].rolling(w, min_periods=w).mean()
        out["lagged_variance"] = out["lagged_budget"].rolling(w, min_periods=w).var()
        out["covariance"] = out["budget"].rolling(w, min_periods=w).cov(out["lagged_budget"])
        out["beta"] = out["covariance"] / out["lagged_variance"].where(out["lagged_variance"] > 1e-12)
        out["forecast"] = (out["budget_mean"] + out["beta"] * (out["budget"] - out["lagged_mean"])) / h
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapBudgetParams) -> SignalFrame:
        position = 0
        streak = 0
        signals = np.zeros(len(data), dtype=int)
        for i, forecast in enumerate(indicators["forecast"].to_numpy(dtype=float)):
            if not np.isfinite(forecast):
                streak = 0
            elif forecast > params.entry_gap:
                streak += 1
                if streak >= 2:
                    position = 1
            else:
                streak = 0
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"forecast": [np.nan, 0.002, 0.003, np.nan, -0.001, 0.002, 0.003, -0.001, 0.002]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, GapBudgetParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1, 0]
    assert (result.data["size"] > 0).all()
