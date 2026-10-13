from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecurrenceParams:
    pattern_window: int = 8
    trend_window: int = 8
    complexity_ceiling: float = 0.65
    efficiency_floor: float = 0.35


class GeneratedStrategy(BaseStrategy[OrdinalRecurrenceParams]):
    strategy_id = "gen_local-w2_1791194709"

    @classmethod
    def params_type(cls) -> type[OrdinalRecurrenceParams]:
        return OrdinalRecurrenceParams

    @staticmethod
    def warmup_bars(params: OrdinalRecurrenceParams) -> int:
        return int(max(params.pattern_window + 6, params.trend_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalRecurrenceParams) -> pd.DataFrame:
        if params.pattern_window < 1 or params.trend_window < 1:
            raise ValueError("Lookback windows must be positive.")
        if not 0.0 <= params.complexity_ceiling <= 1.0:
            raise ValueError("complexity_ceiling must be in [0, 1].")
        if not 0.0 < params.efficiency_floor <= 1.0:
            raise ValueError("efficiency_floor must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out["return"] = (close / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack((out["return"].shift(2), out["return"].shift(1), out["return"]))
        # Stable ordering breaks ties by observation time without future information.
        order = np.argsort(triples, axis=1, kind="stable")
        code = pd.Series(order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2], index=data.index, dtype=float)
        out["ordinal_code"] = code.where(np.isfinite(triples).all(axis=1))
        previous = out["ordinal_code"].shift(3)
        valid = out["ordinal_code"].notna() & previous.notna()
        out["recurrence"] = out["ordinal_code"].eq(previous).astype(float).where(valid)
        # Compare disjoint return triples; complexity measures failure to repeat a motif.
        out["ordinal_complexity"] = 1.0 - out["recurrence"].rolling(
            params.pattern_window, min_periods=params.pattern_window
        ).mean()
        out["absolute_step"] = close.diff().abs()
        out["path_length"] = out["absolute_step"].rolling(
            params.trend_window, min_periods=params.trend_window
        ).sum()
        out["displacement"] = close.diff(params.trend_window)
        out["directional_efficiency"] = (
            out["displacement"] / out["path_length"].where(out["path_length"].gt(0))
        ).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalRecurrenceParams,
    ) -> SignalFrame:
        ready = indicators[["ordinal_complexity", "directional_efficiency"]].notna().all(axis=1)
        entry = (
            ready
            & indicators["ordinal_complexity"].le(params.complexity_ceiling)
            & indicators["directional_efficiency"].ge(params.efficiency_floor)
        )
        df = pd.DataFrame(index=data.index)
        # Maintain the long target only while both entry primitives agree.
        # The first false joint predicate exits; no independent timed or price exit.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = OrdinalRecurrenceParams()
    t = np.arange(40, dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.00001 * t * t)
    data = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(t)),
    )
    indicators = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].iloc[:GeneratedStrategy.warmup_bars(params)].eq(0).all()
    assert signals["signal"].iloc[14:].eq(1).all()
    indicators.loc[data.index[25], "directional_efficiency"] = -1.0
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].iloc[24:28].tolist() == [1, 1, 0, 1]
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
