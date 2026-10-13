from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    complexity_window: int = 10
    complexity_ceiling: float = 0.85


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = "gen_local-w2_1791311433"

    @classmethod
    def params_type(cls) -> type[TideParams]:
        return TideParams

    def warmup_bars(self, params: TideParams) -> int:
        return int(params.complexity_window) + 2

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        if params.complexity_window < 3:
            raise ValueError("complexity_window must be at least 3")
        if not 0.0 <= params.complexity_ceiling <= 1.0:
            raise ValueError("complexity_ceiling must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        out["intraday_return"] = (
            data["close"] / data["open"].where(data["open"] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        values = np.column_stack([
            out["intraday_return"].shift(2).to_numpy(),
            out["intraday_return"].shift(1).to_numpy(),
            out["intraday_return"].to_numpy(),
        ])
        valid = pd.Series(np.isfinite(values).all(axis=1), index=data.index)
        # Stable ordering assigns equal observations by their temporal order.
        order = np.argsort(values, axis=1, kind="stable")
        out["ordinal_pattern"] = pd.Series(
            order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2],
            index=data.index,
            dtype=float,
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (5, 7, 11, 15, 19, 21):
            probability = (
                out["ordinal_pattern"].eq(code).astype(float).where(valid)
                .rolling(params.complexity_window, min_periods=params.complexity_window)
                .mean()
            )
            out[f"pattern_probability_{code}"] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out["complexity"] = (entropy / np.log(6.0)).clip(0.0, 1.0)
        out["calendar_phase"] = np.where(data.index.day <= 15, 1, -1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideParams,
    ) -> SignalFrame:
        qualified = (
            indicators["complexity"].notna()
            & indicators["complexity"].le(params.complexity_ceiling)
        )
        desired = indicators["calendar_phase"].where(qualified, 0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired.to_numpy()):
            if direction != 0:
                if position == -direction:
                    position = 0
                elif position == 0:
                    position = int(direction)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-13", periods=5, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "complexity": [0.2, np.nan, 0.2, 0.2, 0.2],
        "calendar_phase": [1, -1, -1, -1, 1],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, TideParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
