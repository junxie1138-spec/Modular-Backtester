from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalChangeParams:
    recent_window: int = 10
    baseline_window: int = 40
    complexity_drop: float = 0.12
    ascending_advantage: float = 0.15


class GeneratedStrategy(BaseStrategy[OrdinalChangeParams]):
    strategy_id = "gen_local_1791293189"

    @classmethod
    def params_type(cls) -> type[OrdinalChangeParams]:
        return OrdinalChangeParams

    def warmup_bars(self, params: OrdinalChangeParams) -> int:
        return int(params.recent_window + params.baseline_window + 3)

    def indicators(self, data: pd.DataFrame, params: OrdinalChangeParams) -> pd.DataFrame:
        if params.recent_window < 2 or params.baseline_window < 2:
            raise ValueError("Ordinal windows must be at least two bars")
        if not 0.0 <= params.complexity_drop <= 1.0:
            raise ValueError("complexity_drop must be between zero and one")
        if not 0.0 <= params.ascending_advantage <= 1.0:
            raise ValueError("ascending_advantage must be between zero and one")
        close = data["close"].where(np.isfinite(data["close"]))
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Equal prices are ordered chronologically, giving deterministic ties.
        pattern = (
            4 * a.le(b).astype(int)
            + 2 * a.le(c).astype(int)
            + b.le(c).astype(int)
        ).astype(float).where(valid)
        out = pd.DataFrame(index=data.index)
        out["ordinal_pattern"] = pattern
        codes = (0, 1, 3, 4, 6, 7)
        recent_columns, baseline_columns = [], []
        for code in codes:
            occupied = pattern.eq(code).astype(float).where(valid)
            recent_name = f"recent_p_{code}"
            baseline_name = f"baseline_p_{code}"
            out[recent_name] = occupied.rolling(
                params.recent_window, min_periods=params.recent_window
            ).mean()
            out[baseline_name] = occupied.shift(params.recent_window).rolling(
                params.baseline_window, min_periods=params.baseline_window
            ).mean()
            recent_columns.append(recent_name)
            baseline_columns.append(baseline_name)
        out["recent_complexity"] = (
            1.0 - out[recent_columns].pow(2).sum(axis=1, min_count=6)
        ) / (5.0 / 6.0)
        out["baseline_complexity"] = (
            1.0 - out[baseline_columns].pow(2).sum(axis=1, min_count=6)
        ) / (5.0 / 6.0)
        out["complexity_decline"] = out["baseline_complexity"] - out["recent_complexity"]
        out["ascending_advantage"] = out["recent_p_7"] - out["recent_p_0"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalChangeParams,
    ) -> SignalFrame:
        eligible = (
            indicators["complexity_decline"].ge(params.complexity_drop)
            & indicators["ascending_advantage"].ge(params.ascending_advantage)
        ).fillna(False)
        # Two consecutive eligible closes enter; the first ineligible close exits.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame(index=pd.date_range("2020-01-01", periods=7))
    derived = pd.DataFrame({
        "complexity_decline": [0.3, 0.3, 0.3, 0.0, 0.3, 0.3, np.nan],
        "ascending_advantage": [0.4] * 7,
    }, index=sample.index)
    result = GeneratedStrategy.generate_signals(
        None, sample, derived, None, OrdinalChangeParams()
    )
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert result.data["size"].gt(0).all()
