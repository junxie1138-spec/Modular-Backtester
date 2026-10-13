from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapComplexityParams:
    pattern_window: int = 20
    min_complexity: float = 0.06


class GeneratedStrategy(BaseStrategy[GapComplexityParams]):
    strategy_id = "gen_local-w2_1791319235"

    @classmethod
    def params_type(cls) -> type[GapComplexityParams]:
        return GapComplexityParams

    @staticmethod
    def warmup_bars(params: GapComplexityParams) -> int:
        return int(params.pattern_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapComplexityParams) -> pd.DataFrame:
        if params.pattern_window < 3 or not 0.0 <= params.min_complexity <= 1.0:
            raise ValueError("Require pattern_window >= 3 and min_complexity in [0, 1].")
        out = pd.DataFrame(index=data.index)
        a = data["close"].shift(1)
        b = data["open"]
        c = data["close"]
        out["previous_close"] = a
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        valid &= (a > 0) & (b > 0) & (c > 0)
        valid &= a.ne(b) & a.ne(c) & b.ne(c)
        out["gap"] = (b / a - 1.0).where(valid)
        out["recovery"] = (c / a - 1.0).where(valid)
        # Six strict ordinal motifs of previous close, current open, current close.
        out["pattern"] = (
            4 * a.lt(b).astype(int)
            + 2 * a.lt(c).astype(int)
            + b.lt(c).astype(int)
        ).astype(float).where(valid)
        probability_columns = []
        for motif in (0, 1, 3, 4, 6, 7):
            name = f"p_{motif}"
            out[name] = out["pattern"].eq(motif).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        entropy_terms = probabilities * np.log(probabilities.clip(lower=1e-12))
        out["entropy"] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = (
            probabilities.sub(1.0 / 6.0).pow(2).sum(axis=1, min_count=6) * 6.0 / 5.0
        )
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapComplexityParams,
    ) -> SignalFrame:
        condition = (
            indicators["gap"].lt(0.0)
            & indicators["recovery"].gt(0.0)
            & indicators["complexity"].ge(params.min_complexity)
        ).fillna(False)
        # Enter on the second consecutive qualifying close; exit on its first reversal.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapComplexityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    probe = pd.DataFrame({
        "gap": [-0.01] * 8,
        "recovery": [0.01, 0.01, 0.01, -0.01, 0.01, 0.01, -0.01, -0.01],
        "complexity": [0.1] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 0]
    assert result["size"].gt(0.0).all()
