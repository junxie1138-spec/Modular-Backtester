from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShiftParams:
    recent_bars: int = 8
    baseline_bars: int = 32
    shift_threshold: float = 1.0


class GeneratedStrategy(BaseStrategy[GapShiftParams]):
    strategy_id = "gen_local-w2_1791246423"

    @classmethod
    def params_type(cls) -> type[GapShiftParams]:
        return GapShiftParams

    def warmup_bars(self, params: GapShiftParams) -> int:
        return int(max(200, params.recent_bars + params.baseline_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapShiftParams) -> pd.DataFrame:
        if params.recent_bars < 2 or params.baseline_bars < 2:
            raise ValueError("Gap windows must contain at least two bars")
        if not np.isfinite(params.shift_threshold) or params.shift_threshold <= 0:
            raise ValueError("shift_threshold must be finite and positive")

        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"].div(prior_close) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["older_gap"] = out["gap"].shift(params.recent_bars)
        recent = out["gap"].rolling(params.recent_bars, min_periods=params.recent_bars)
        older = out["older_gap"].rolling(params.baseline_bars, min_periods=params.baseline_bars)
        out["recent_mean"] = recent.mean()
        out["baseline_mean"] = older.mean()
        out["recent_variance"] = recent.var(ddof=1)
        out["baseline_variance"] = older.var(ddof=1)
        out["mean_shift"] = out["recent_mean"] - out["baseline_mean"]
        out["standard_error"] = np.sqrt(
            out["recent_variance"] / params.recent_bars
            + out["baseline_variance"] / params.baseline_bars
        ).clip(lower=1e-8)
        out["shift_score"] = out["mean_shift"] / out["standard_error"]
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShiftParams,
    ) -> SignalFrame:
        entry_condition = (
            (indicators["shift_score"] > params.shift_threshold)
            & (indicators["recent_mean"] > 0.0)
            & (data["close"] >= indicators["ma200"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; exit only when it flips false.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapShiftParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=340, freq="B")
    gaps = np.zeros(len(index))
    gaps[220:260] = 0.01
    gaps[260:300] = -0.01
    close = np.full(len(index), 100.0)
    opening = close * (1.0 + gaps)
    data = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) + 1.0,
        "low": np.minimum(opening, close) - 1.0,
        "close": close,
        "volume": np.full(len(index), 1000000.0),
    }, index=index)
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    condition = (
        (derived["shift_score"] > params.shift_threshold)
        & (derived["recent_mean"] > 0.0)
        & (data["close"] >= derived["ma200"])
    ).fillna(False)
    expected = condition.astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[220:260].eq(1).any()
    assert result["signal"].iloc[270:300].eq(0).all()
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= max(200, params.recent_bars + params.baseline_bars + 1)
    prefix = data.iloc[:250]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert prefix_result.equals(result.iloc[:250])


if __name__ == "__main__":
    _self_check()
