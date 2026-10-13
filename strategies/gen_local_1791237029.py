from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockParams:
    memory_bars: int = 18
    percentile_bars: int = 126
    tail_percentile: float = 0.20


class GeneratedStrategy(BaseStrategy[GapShockParams]):
    strategy_id = "gen_local_1791237029"

    @classmethod
    def params_type(cls) -> type[GapShockParams]:
        return GapShockParams

    @staticmethod
    def warmup_bars(params: GapShockParams) -> int:
        return int(params.memory_bars + params.percentile_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapShockParams) -> pd.DataFrame:
        if params.memory_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.tail_percentile < 0.5:
            raise ValueError("tail_percentile must be between zero and one half")

        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (
            data["open"].div(out["prior_close"].where(out["prior_close"] > 0.0)) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["gap_cube"] = out["gap"].pow(3).replace([np.inf, -np.inf], np.nan)
        out["absolute_cube"] = out["gap_cube"].abs()
        out["signed_shock"] = out["gap_cube"].rolling(
            params.memory_bars, min_periods=params.memory_bars
        ).sum()
        out["total_shock"] = out["absolute_cube"].rolling(
            params.memory_bars, min_periods=params.memory_bars
        ).sum()
        out["balance"] = out["signed_shock"].div(
            out["total_shock"].where(out["total_shock"] > 0.0)
        )
        # Thresholds exclude the current observation.
        out["historical_balance"] = out["balance"].shift(1)
        history = out["historical_balance"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        )
        out["lower"] = history.quantile(params.tail_percentile)
        out["upper"] = history.quantile(1.0 - params.tail_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShockParams,
    ) -> SignalFrame:
        balance = indicators["balance"]
        valid = np.isfinite(indicators[["balance", "lower", "upper"]]).all(axis=1)
        entry = valid & (balance < 0.0) & (balance <= indicators["lower"])
        reversal = valid & (balance > 0.0) & (balance >= indicators["upper"])

        # Neutral observations retain the position; only opposite shocks exit.
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[entry] = 1.0
        decisions.loc[reversal] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {
            "balance": [np.nan, -0.9, -0.1, 0.9, -0.9, 0.0],
            "lower": -0.8,
            "upper": 0.8,
        },
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapShockParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert result.data["size"].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(GapShockParams()) == 145
    derived = GeneratedStrategy.indicators(data, GapShockParams())
    assert derived.index.equals(data.index)
    assert derived["balance"].isna().all()


if __name__ == "__main__":
    _self_check()
