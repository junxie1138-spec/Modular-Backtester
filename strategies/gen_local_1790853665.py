from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalOrdinalParams:
    pattern_window: int = 6
    max_patterns: int = 3
    season_start: int = 8
    season_end: int = 20


class GeneratedStrategy(BaseStrategy[SeasonalOrdinalParams]):
    strategy_id = "gen_local_1790853665"

    @classmethod
    def params_type(cls) -> type[SeasonalOrdinalParams]:
        return SeasonalOrdinalParams

    def warmup_bars(self, params: SeasonalOrdinalParams) -> int:
        if not 2 <= params.pattern_window <= 8:
            raise ValueError("pattern_window must be between 2 and 8")
        if not 1 <= params.max_patterns <= 6:
            raise ValueError("max_patterns must be between 1 and 6")
        if not 1 <= params.season_start < params.season_end <= 28:
            raise ValueError("season dates must satisfy 1 <= start < end <= 28")
        return int(params.pattern_window + 2)

    def indicators(
        self, data: pd.DataFrame, params: SeasonalOrdinalParams
    ) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        oldest, middle = close.shift(2), close.shift(1)
        valid = oldest.notna() & middle.notna() & close.notna()
        # Equal prices retain chronological order, giving six possible motifs.
        out["pattern"] = (
            4 * oldest.le(middle).astype(int)
            + 2 * oldest.le(close).astype(int)
            + middle.le(close).astype(int)
        ).where(valid)
        count_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = f"count_{code}"
            out[column] = (
                out["pattern"].eq(code).astype(float).where(valid)
                .rolling(params.pattern_window, min_periods=params.pattern_window).sum()
            )
            count_columns.append(column)
        counts = out[count_columns]
        out["complexity"] = (
            counts.gt(0).sum(axis=1).astype(float)
            .where(counts.notna().all(axis=1))
        )
        out["drift"] = (
            close / close.shift(params.pattern_window + 1) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            out["complexity"].le(params.max_patterns) & out["drift"].gt(0)
        ).fillna(False)
        out["season"] = (
            (data.index.day >= params.season_start)
            & (data.index.day < params.season_end)
        )
        out["month"] = data.index.year * 12 + data.index.month
        out["launch"] = out["season"] & (
            ~out["season"].shift(1, fill_value=False)
            | out["month"].ne(out["month"].shift(1))
        )
        # Sample once per seasonal episode; no future observations enter admission.
        admission = (
            out["eligible"].astype(float).where(out["launch"])
            .groupby(out["month"]).ffill().fillna(0.0)
        )
        out["admitted"] = admission.gt(0) & out["season"]
        out["entry_condition"] = out["season"] & out["admitted"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalOrdinalParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Reversal exit: flat precisely when the seasonal entry predicate flips off.
        df["signal"] = indicators["entry_condition"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SeasonalOrdinalParams()
    index = pd.bdate_range("2024-01-22", periods=60)
    close = pd.Series(100.0 + np.arange(len(index)), index=index)
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000000.0},
        index=index,
    )
    features = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, features, None, params).data
    expected = pd.Series(
        ((index.day >= 8) & (index.day < 20)).astype(int), index=index
    ).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["size"].gt(0).all()
    assert strategy.warmup_bars(SeasonalOrdinalParams(pattern_window=8)) == 10
    pd.testing.assert_frame_equal(
        features.iloc[:20], strategy.indicators(data.iloc[:20], params)
    )
    missing = data.copy()
    missing["close"] = np.nan
    assert not strategy.generate_signals(
        missing, strategy.indicators(missing, params), None, params
    ).data["signal"].any()
