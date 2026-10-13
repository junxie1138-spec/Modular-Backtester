from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionTraversalParams:
    rank_window: int = 40
    compression_percentile: float = 0.30
    auction_tail: float = 0.25
    atr_window: int = 14
    stop_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionTraversalParams]):
    strategy_id = "gen_local-w2_1790998268"

    @classmethod
    def params_type(cls):
        return AuctionTraversalParams

    def warmup_bars(self, params: AuctionTraversalParams) -> int:
        return int(max(params.rank_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: AuctionTraversalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["normalized_range"] = out["true_range"] / out["previous_close"].where(out["previous_close"] > 0)
        out["compression_rank"] = out["normalized_range"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        denominator = out["daily_range"].where(out["daily_range"] > 0)
        out["open_location"] = ((data["open"] - data["low"]) / denominator).clip(0.0, 1.0)
        out["close_location"] = ((data["close"] - data["low"]) / denominator).clip(0.0, 1.0)
        out["open_rank"] = out["open_location"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["close_rank"] = out["close_location"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionTraversalParams,
    ) -> SignalFrame:
        valid = indicators[["compression_rank", "open_rank", "close_rank", "atr"]].notna().all(axis=1)
        valid &= indicators["atr"] > 0
        compact = indicators["compression_rank"] <= params.compression_percentile
        # Primitive one: an unusually low/high opening location.
        opening_long = indicators["open_rank"] <= params.auction_tail
        opening_short = indicators["open_rank"] >= 1.0 - params.auction_tail
        # Primitive two: independent agreement from closing acceptance.
        closing_long = indicators["close_rank"] >= 1.0 - params.auction_tail
        closing_short = indicators["close_rank"] <= params.auction_tail
        long_entry = valid & compact & opening_long & closing_long & (data["close"] > data["open"])
        short_entry = valid & compact & opening_short & closing_short & (data["close"] < data["open"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        frozen_atr = np.nan
        horizon = min(2, max(1, int(params.hold_bars)))
        for i in range(len(data)):
            if direction != 0:
                # A decision at the preceding close fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                stopped = False
                if not invalid:
                    if direction == 1:
                        stopped = closes[i] < entry_price - params.stop_atr * frozen_atr
                    else:
                        stopped = closes[i] > entry_price + params.stop_atr * frozen_atr
                expired = i - entry_bar + 1 >= horizon
                if invalid or stopped or expired:
                    direction = 0
                    entry_price = np.nan
                    frozen_atr = np.nan
                    continue
                decisions[i] = direction
            elif candidates[i] != 0:
                direction = int(candidates[i])
                entry_bar = i + 1
                frozen_atr = atrs[i]
                decisions[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
