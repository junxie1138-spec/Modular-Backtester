from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 63
    block_window: int = 8
    shift_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791346011"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.rank_window + 2 * params.block_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.rank_window < 2 or params.block_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not np.isfinite(params.shift_threshold) or not 0 < params.shift_threshold < 1:
            raise ValueError("shift_threshold must be finite and between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_percentile"] = out["return"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["recent_percentile"] = out["return_percentile"].rolling(
            params.block_window, min_periods=params.block_window
        ).mean()
        out["previous_percentile"] = out["recent_percentile"].shift(params.block_window)
        out["percentile_shift"] = out["recent_percentile"] - out["previous_percentile"]
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        valid = indicators[["recent_percentile", "percentile_shift", "ma_200"]].notna().all(axis=1)
        entry_condition = (
            valid
            & (data["close"] > indicators["ma_200"])
            & (indicators["recent_percentile"] > 0.5)
            & (indicators["percentile_shift"] > params.shift_threshold)
        )
        df = pd.DataFrame(index=data.index)
        # Reversal exit: the position becomes flat whenever the entry condition flips false.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"close": [100.0] * 4}, index=index)
    indicators = pd.DataFrame(
        {
            "recent_percentile": [0.7] * 4,
            "percentile_shift": [np.nan, 0.2, -0.1, 0.2],
            "ma_200": [90.0] * 4,
        },
        index=index,
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(Params(rank_window=220)) == 236


if __name__ == "__main__":
    _self_check()
