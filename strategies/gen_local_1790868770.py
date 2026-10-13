from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_bars: int = 30
    residence_bars: int = 5
    depth_threshold: float = 0.012
    min_residence: float = 0.8
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790868770"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.peak_bars + params.residence_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.peak_bars < 2 or params.residence_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0 < params.depth_threshold < 1 or not 0.5 < params.min_residence <= 1:
            raise ValueError("Invalid depth or residence threshold.")
        if params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Positive target and one- or two-bar hold required.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["prior_depth"] = out["depth"].shift(1)
        out["depth_change"] = out["depth"].diff()
        out["deep"] = (out["depth"] >= params.depth_threshold).astype(float).where(out["depth"].notna())
        out["residence"] = out["deep"].shift(1).rolling(
            params.residence_bars, min_periods=params.residence_bars
        ).mean()
        # A rolling peak expiring must not masquerade as a price recovery.
        out["stable_peak"] = out["peak"].eq(out["peak"].shift(1))
        out["ready"] = out["residence"].notna() & out["depth"].notna() & out["stable_peak"]
        long_entry = (
            out["ready"]
            & (out["residence"] >= params.min_residence)
            & (out["prior_depth"] >= params.depth_threshold)
            & (out["depth"] < params.depth_threshold)
        )
        short_entry = (
            out["ready"]
            & (out["residence"] <= 1.0 - params.min_residence + 1e-12)
            & (out["prior_depth"] < params.depth_threshold)
            & (out["depth"] >= params.depth_threshold)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        out["strength"] = (out["depth_change"].abs() / params.depth_threshold).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        strengths = indicators["strength"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position, entry_bar = 0, -1
        entry_price, frozen_size = np.nan, 0.25
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                frozen_size = 0.25 + 0.70 * float(np.clip(strengths[i], 0.0, 1.0))
            decisions[i] = position
            if position:
                sizes[i] = frozen_size
        df["signal"] = decisions
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0,
         "close": [100.0, 100.9, 100.0, 99.1, 100.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"entry": [1, 0, -1, 0, 1, 0, 0, 0], "strength": 0.5}, index=index
    )
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, -1, 0, 1, 1, 0]
    assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.60)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
