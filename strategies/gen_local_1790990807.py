from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PersistenceParams:
    recent_window: int = 5
    baseline_window: int = 20
    rank_window: int = 126
    entry_percentile: float = 0.85
    min_up_share: float = 0.60
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[PersistenceParams]):
    strategy_id = "gen_local_1790990807"

    @classmethod
    def params_type(cls) -> type[PersistenceParams]:
        return PersistenceParams

    def warmup_bars(self, params: PersistenceParams) -> int:
        return int(max(
            params.recent_window + params.baseline_window + params.rank_window + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: PersistenceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff()
        out["direction"] = np.sign(out["change"])
        out["concordance"] = out["direction"] * out["direction"].shift(1)
        out["recent_concordance"] = out["concordance"].rolling(
            params.recent_window, min_periods=params.recent_window
        ).mean()
        out["baseline_concordance"] = out["concordance"].shift(
            params.recent_window
        ).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["persistence_jump"] = (
            out["recent_concordance"] - out["baseline_concordance"]
        )
        out["jump_percentile"] = out["persistence_jump"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["up_day"] = out["direction"].gt(0).astype(float).where(
            out["direction"].notna()
        )
        out["up_share"] = out["up_day"].rolling(
            params.recent_window, min_periods=params.recent_window
        ).mean()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["candidate"] = (
            out["jump_percentile"].ge(params.entry_percentile)
            & out["baseline_concordance"].le(0.0)
            & out["recent_concordance"].ge(0.2)
            & out["up_share"].ge(params.min_up_share)
            & out["atr"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = (
            out["candidate"] & out["candidate"].shift(1, fill_value=False)
        )
        out["entry_event"] = (
            out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PersistenceParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        target = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].to_numpy(dtype=bool)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0.0):
                active = False
                held = 0
                high_water = -np.inf
                stop = -np.inf
                continue
            if active:
                # First update is the close of the next-bar entry session.
                held += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or held >= params.max_hold_bars:
                    active = False
            elif entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
            target[i] = int(active)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
