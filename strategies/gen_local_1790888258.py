from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBoundaryParams:
    entry_z: float = 1.5
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[GapBoundaryParams]):
    strategy_id = "gen_local_1790888258"

    @classmethod
    def params_type(cls) -> type[GapBoundaryParams]:
        return GapBoundaryParams

    def warmup_bars(self, params: GapBoundaryParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: GapBoundaryParams) -> pd.DataFrame:
        prior_close = data["close"].shift(1)
        mean = prior_close.rolling(20, min_periods=20).mean()
        std = prior_close.rolling(20, min_periods=20).std(ddof=0)
        std = std.where(std > 0)
        result = pd.DataFrame(index=data.index)
        result["prior_close"] = prior_close
        result["mean"] = mean
        result["std"] = std
        result["open_z"] = (data["open"] - mean) / std
        result["close_z"] = (data["close"] - mean) / std
        result["gap"] = data["open"] / prior_close.where(prior_close > 0) - 1.0
        result["intraday_return"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapBoundaryParams) -> SignalFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")

        valid = indicators.notna().all(axis=1)
        boundary = params.entry_z
        reentry_boundary = 0.75 * boundary
        longs = (
            valid
            & (indicators["gap"] < 0)
            & (indicators["open_z"] <= -boundary)
            & (indicators["close_z"] >= -reentry_boundary)
            & (indicators["close_z"] < 0)
            & (indicators["intraday_return"] > 0)
            & (data["close"] < indicators["prior_close"])
        )
        shorts = (
            valid
            & (indicators["gap"] > 0)
            & (indicators["open_z"] >= boundary)
            & (indicators["close_z"] <= reentry_boundary)
            & (indicators["close_z"] > 0)
            & (indicators["intraday_return"] < 0)
            & (data["close"] > indicators["prior_close"])
        )
        entries = np.select([longs, shorts], [1, -1], default=0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        held_bars = 0

        for i in range(len(data)):
            # Yesterday's decision fills at today's open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = int(decisions[i - 1])
                    entry_price = opens[i]
                    held_bars = 0
                else:
                    continue

            if position != 0:
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                # Close-based exits execute at the following open.
                if target_hit or held_bars >= 2:
                    decisions[i] = 0
                    position = 0
                    held_bars = 0
                else:
                    decisions[i] = position
            else:
                decisions[i] = entries[i]

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
