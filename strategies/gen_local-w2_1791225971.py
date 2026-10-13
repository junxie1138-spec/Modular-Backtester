from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionReleaseParams:
    breakout_bars: int = 25
    rank_window: int = 80
    rejection_days: int = 5
    rejection_threshold: float = 0.75
    wick_threshold: float = 0.40
    atr_bars: int = 14
    stop_atr: float = 2.8
    max_hold: int = 23


class GeneratedStrategy(BaseStrategy[AuctionReleaseParams]):
    strategy_id = "gen_local-w2_1791225971"

    @classmethod
    def params_type(cls) -> type[AuctionReleaseParams]:
        return AuctionReleaseParams

    @classmethod
    def warmup_bars(cls, params: AuctionReleaseParams) -> int:
        return max(params.rank_window + params.rejection_days + 1,
                   params.breakout_bars + 1, params.atr_bars)

    def indicators(self, data: pd.DataFrame, params: AuctionReleaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        bar_range = (data["high"] - data["low"]).replace(0, np.nan)
        upper_wick = ((data["high"] - data[["open", "close"]].max(axis=1)) / bar_range).fillna(0.0)
        prior_rejection = upper_wick.rolling(params.rejection_days, min_periods=params.rejection_days).mean().shift(1)
        out["rejection_rank"] = prior_rejection.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True).fillna(0.0)
        out["wick_rank"] = upper_wick.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True).fillna(1.0)
        out["prior_high"] = data["high"].rolling(params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([data["high"] - data["low"],
                                (data["high"] - previous_close).abs(),
                                (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AuctionReleaseParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.5
        eligible = ((data["close"] > indicators["prior_high"])
                    & (data["close"] > data["open"])
                    & (indicators["rejection_rank"] >= params.rejection_threshold)
                    & (indicators["wick_rank"] <= params.wick_threshold)
                    & (indicators["atr"] > 0)).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        rejection_ranks = indicators["rejection_rank"].to_numpy(dtype=float)
        wick_ranks = indicators["wick_rank"].to_numpy(dtype=float)
        in_trade = False
        high_water = 0.0
        stop = 0.0
        held = 0
        entry_size = 0.5
        for i in range(len(df)):
            if in_trade:
                high_water = max(high_water, closes[i]) if held else closes[i]
                stop = max(stop, high_water - params.stop_atr * atrs[i]) if held else high_water - params.stop_atr * atrs[i]
                held += 1
                if closes[i] <= stop or held >= params.max_hold:
                    in_trade = False
                else:
                    df.iat[i, 0] = 1
                    df.iat[i, 1] = entry_size
            elif eligible[i]:
                in_trade = True
                held = 0
                entry_size = float(np.clip(0.35 + 0.6 * (rejection_ranks[i] - wick_ranks[i]), 0.35, 0.95))
                df.iat[i, 0] = 1
                df.iat[i, 1] = entry_size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
