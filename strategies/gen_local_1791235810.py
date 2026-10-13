from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelocationParams:
    range_bars: int = 20
    rank_bars: int = 60
    rank_threshold: float = 0.90
    holding_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RelocationParams]):
    strategy_id = "gen_local_1791235810"

    @classmethod
    def params_type(cls) -> type[RelocationParams]:
        return RelocationParams

    def warmup_bars(self, params: RelocationParams) -> int:
        if params.range_bars < 2 or params.rank_bars < 2 or params.holding_bars < 1:
            raise ValueError("Lookbacks must be >= 2 and holding_bars >= 1")
        if not 0.0 < params.rank_threshold < 1.0:
            raise ValueError("rank_threshold must be between zero and one")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        return int(params.range_bars + params.rank_bars)

    def indicators(self, data: pd.DataFrame, params: RelocationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["ceiling"] = data["high"].rolling(params.range_bars).max().shift(1)
        out["floor"] = data["low"].rolling(params.range_bars).min().shift(1)
        out["daily_range"] = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["upper_share"] = (
            (data["high"] - np.maximum(data["low"], out["ceiling"]))
            / out["daily_range"]
        ).clip(0.0, 1.0)
        out["lower_share"] = (
            (np.minimum(data["high"], out["floor"]) - data["low"])
            / out["daily_range"]
        ).clip(0.0, 1.0)
        out["relocation"] = out[["upper_share", "lower_share"]].max(axis=1, skipna=False)
        out["relocation_rank"] = out["relocation"].rolling(params.rank_bars).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelocationParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ranked = indicators["relocation_rank"]
        eligible = ranked.ge(params.rank_threshold) & indicators["relocation"].gt(0.5)
        long_entry = eligible & data["close"].gt(indicators["ceiling"]) & indicators["upper_share"].gt(0.5)
        short_entry = eligible & data["close"].lt(indicators["floor"]) & indicators["lower_share"].gt(0.5)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        strength = (
            (ranked - params.rank_threshold) / (1.0 - params.rank_threshold)
        ).clip(0.0, 1.0).fillna(0.0)
        entry_sizes = (
            params.min_size
            + (params.max_size - params.min_size) * strength * indicators["relocation"].fillna(0.0)
        ).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if position != 0 and i == exit_bar:
                position = 0
                held_size = params.min_size
                # Emit a flat bar; do not reopen on the expiry decision.
            elif position == 0 and candidates[i] != 0:
                position = int(candidates[i])
                exit_bar = i + params.holding_bars
                held_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RelocationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0},
        index=index,
    )
    derived = pd.DataFrame(
        {"ceiling": 11.0, "floor": 9.0, "upper_share": 0.0,
         "lower_share": 0.0, "relocation": 0.0, "relocation_rank": 0.99},
        index=index,
    )
    derived.loc[index[0], ["ceiling", "upper_share", "relocation"]] = [9.0, 0.8, 0.8]
    derived.loc[index[[2, 7]], ["floor", "lower_share", "relocation"]] = [11.0, 0.8, 0.8]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.754)
    assert result["size"].iloc[1:5].nunique() == 1
    assert result["size"].gt(0.0).all()
    quiet = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert quiet["signal"].eq(0).all()
