from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumFloorParams:
    week_bars: int = 5
    history_bars: int = 252
    entry_quantile: float = 0.65


class GeneratedStrategy(BaseStrategy[MomentumFloorParams]):
    strategy_id = "gen_local-w2_1790966786"

    @classmethod
    def params_type(cls) -> type[MomentumFloorParams]:
        return MomentumFloorParams

    @staticmethod
    def warmup_bars(params: MomentumFloorParams) -> int:
        return int(4 * params.week_bars + params.history_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: MomentumFloorParams) -> pd.DataFrame:
        if params.week_bars < 1 or params.history_bars < 2:
            raise ValueError("Lookbacks must be positive, with history_bars >= 2")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must lie strictly between zero and one")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["week_0"] = out["return"].rolling(params.week_bars, min_periods=params.week_bars).sum()
        for block in range(1, 4):
            out[f"week_{block}"] = out["week_0"].shift(block * params.week_bars)
        out["momentum_floor"] = out[["week_0", "week_1", "week_2", "week_3"]].min(axis=1, skipna=False)
        out["threshold"] = out["momentum_floor"].shift(1).rolling(params.history_bars, min_periods=params.history_bars).quantile(params.entry_quantile)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumFloorParams) -> SignalFrame:
        qualified = (
            indicators["momentum_floor"].notna()
            & indicators["threshold"].notna()
            & indicators["momentum_floor"].gt(0.0)
            & indicators["momentum_floor"].gt(indicators["threshold"])
        )
        df = pd.DataFrame(index=data.index)
        # A held position exits exactly when the same entry condition becomes false.
        df["signal"] = qualified.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = MomentumFloorParams()
    index = pd.date_range("2020-01-01", periods=700, freq="B")
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.003 * np.sin(np.arange(700) / 13.0)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected = (derived["momentum_floor"].gt(0.0) & derived["momentum_floor"].gt(derived["threshold"])).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert set(result["signal"].unique()) == {0, 1}
    assert result["size"].gt(0.0).all()
    assert derived["threshold"].first_valid_index() == index[GeneratedStrategy.warmup_bars(params) - 1]
