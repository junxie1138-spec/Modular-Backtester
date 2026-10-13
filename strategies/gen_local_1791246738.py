from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_window: int = 30
    history_window: int = 126
    shelf_window: int = 8
    deep_percentile: float = 0.80
    shelf_percentile: float = 0.25
    profit_target: float = 0.015
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local_1791246738"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        if min(params.peak_window, params.history_window, params.shelf_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not (0 < params.shelf_percentile < params.deep_percentile < 1):
            raise ValueError("Percentiles must satisfy 0 < shelf < deep < 1")
        if params.profit_target <= 0 or not 3 <= params.holding_bars <= 5:
            raise ValueError("Use a positive target and a three-to-five-session holding period")
        return int(params.peak_window + max(params.history_window, params.shelf_window) + 1)

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"].replace(0.0, np.nan)
        out["prior_depth"] = out["depth"].shift(1)
        out["deep_threshold"] = out["prior_depth"].rolling(
            params.history_window, min_periods=params.history_window
        ).quantile(params.deep_percentile)
        out["shelf"] = out["prior_depth"].rolling(
            params.shelf_window, min_periods=params.shelf_window
        ).quantile(params.shelf_percentile)
        out["prior_shelf"] = out["shelf"].shift(1)
        out["stable_peak"] = out["peak"].eq(out["peak"].shift(1))
        out["rising_close"] = close.gt(close.shift(1))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShelfParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ready = indicators[[
            "depth", "prior_depth", "deep_threshold", "shelf", "prior_shelf"
        ]].notna().all(axis=1)
        entries = (
            ready
            & indicators["prior_depth"].ge(indicators["deep_threshold"])
            & indicators["prior_depth"].gt(0.0)
            & indicators["depth"].lt(indicators["shelf"])
            & indicators["prior_depth"].ge(indicators["prior_shelf"])
            & indicators["stable_peak"]
            & indicators["rising_close"]
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_hit = i - entry_bar + 1 >= params.holding_bars
                if target_hit or time_hit:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                active = True
                entry_bar = i + 1
                entry_price = np.nan

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        assert df["signal"].isin([0, 1]).all() and df["size"].gt(0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
