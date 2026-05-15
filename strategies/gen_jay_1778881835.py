from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveRankParams:
    rank_window: int = 20
    vol_compress_thresh: float = 0.30
    vol_expand_thresh: float = 0.65
    price_rank_thresh: float = 0.60
    profit_target: float = 0.025
    time_stop: int = 5


class GeneratedStrategy(BaseStrategy[ShockwaveRankParams]):
    strategy_id = "gen_jay_1778881835"

    @classmethod
    def params_type(cls) -> type[ShockwaveRankParams]:
        return ShockwaveRankParams

    @staticmethod
    def warmup_bars(params: ShockwaveRankParams) -> int:
        # rank_window for rolling rank + 3 bars consumed by shift chain in confirmation
        return params.rank_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveRankParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        tr_rank = (tr / data["close"]).rolling(params.rank_window).rank(pct=True)
        price_rank = data["close"].rolling(params.rank_window).rank(pct=True)

        # Phase 1 (bar i-1): compressed -> expanding transition
        shockwave_trigger = (tr_rank.shift(1) < params.vol_compress_thresh) & (
            tr_rank > params.vol_expand_thresh
        )
        # Phase 2 (bar i): expansion still holds — two-bar confirmation
        confirmed = shockwave_trigger.shift(1) & (tr_rank > params.vol_expand_thresh)

        ind["tr_rank"] = tr_rank
        ind["price_rank"] = price_rank
        ind["confirmed"] = confirmed.astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveRankParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["close"] = data["close"].values
        df["signal"] = 0
        df["size"] = 1.0

        confirmed = indicators["confirmed"].fillna(0.0).values
        price_rank = indicators["price_rank"].fillna(0.5).values

        raw = np.zeros(len(df), dtype=int)
        long_cond = (confirmed > 0.5) & (price_rank > params.price_rank_thresh)
        short_cond = (confirmed > 0.5) & (
            price_rank < (1.0 - params.price_rank_thresh)
        )
        raw[long_cond] = 1
        raw[short_cond] = -1

        closes = df["close"].values
        out = np.zeros(len(df), dtype=int)
        pos = 0
        entry_px = 0.0
        held = 0

        for i in range(len(df)):
            if pos != 0:
                held += 1
                pnl = (closes[i] - entry_px) / entry_px * pos
                if pnl >= params.profit_target or held >= params.time_stop:
                    pos = 0
                    entry_px = 0.0
                    held = 0

            if pos == 0 and raw[i] != 0:
                pos = raw[i]
                entry_px = closes[i]
                held = 0

            out[i] = pos

        df["signal"] = out
        # Mandatory 1-bar shift: decision at bar N executes at bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
