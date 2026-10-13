from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DistributionShiftParams:
    reaction_window: int = 18
    reference_window: int = 54
    shift_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[DistributionShiftParams]):
    strategy_id = "gen_local-w2_1791251979"

    @classmethod
    def params_type(cls) -> type[DistributionShiftParams]:
        return DistributionShiftParams

    def warmup_bars(self, params: DistributionShiftParams) -> int:
        if params.reaction_window < 2 or params.reference_window < 2:
            raise ValueError("Windows must be at least two bars")
        if not np.isfinite(params.shift_threshold) or params.shift_threshold <= 0:
            raise ValueError("shift_threshold must be finite and positive")
        return int(params.reaction_window + params.reference_window + 1)

    def indicators(self, data: pd.DataFrame, params: DistributionShiftParams) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        recent = returns.rolling(params.reaction_window, min_periods=params.reaction_window)
        reference = returns.shift(params.reaction_window).rolling(
            params.reference_window, min_periods=params.reference_window
        )
        result = pd.DataFrame(index=data.index)
        result["returns"] = returns
        for label, quantile in (("lower", 0.25), ("median", 0.50), ("upper", 0.75)):
            result["recent_" + label] = recent.quantile(quantile)
            result["reference_" + label] = reference.quantile(quantile)
            result["shift_" + label] = result["recent_" + label] - result["reference_" + label]
        result["reference_iqr"] = (
            result["reference_upper"] - result["reference_lower"]
        ).clip(lower=1e-6)
        result["support"] = result[["shift_lower", "shift_median", "shift_upper"]].min(
            axis=1, skipna=False
        ) / result["reference_iqr"]
        total = params.reaction_window + params.reference_window
        result["ready"] = returns.notna().astype(int).rolling(
            total, min_periods=total
        ).sum().eq(total).astype(int)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DistributionShiftParams,
    ) -> SignalFrame:
        active = (
            indicators["ready"].eq(1)
            & indicators["support"].gt(params.shift_threshold)
            & indicators["recent_median"].gt(0.0)
        )
        strength = indicators["support"].clip(lower=0.0)
        conviction = strength / (strength + params.shift_threshold)
        df = pd.DataFrame(index=data.index)
        # Exposure remains active exactly while the entry predicate holds.
        df["signal"] = active.astype(int)
        df["size"] = (0.20 + 0.75 * conviction).clip(lower=0.20, upper=0.95)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DistributionShiftParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "ready": [0, 1, 1, 1, 1, 1],
        "support": [0.0, 0.2, 0.4, 0.0, 0.3, 0.3],
        "recent_median": [0.0, 0.01, 0.01, 0.01, 0.01, -0.01],
    }, index=index)
    frame = strategy.generate_signals(data, indicators, None, params).data
    assert frame["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert frame["size"].between(0.20, 0.95).all()
    assert frame["size"].iloc[3] > frame["size"].iloc[2]
    close = pd.Series(100.0, index=pd.date_range("2021-01-01", periods=100))
    derived = strategy.indicators(pd.DataFrame({"close": close}), params)
    assert np.isfinite(derived.to_numpy()).all()
    assert derived["ready"].iloc[:72].eq(0).all()
    assert derived["ready"].iloc[72:].eq(1).all()
    assert derived["support"].eq(0).all()


if __name__ == "__main__":
    _self_check()
