from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShelfParams:
    gap_atr: float = 0.35
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GapShelfParams]):
    strategy_id = "gen_local_1791236130"

    @classmethod
    def params_type(cls):
        return GapShelfParams

    def warmup_bars(self, params: GapShelfParams) -> int:
        return 18

    def indicators(self, data: pd.DataFrame, params: GapShelfParams) -> pd.DataFrame:
        if not np.isfinite(params.gap_atr) or params.gap_atr <= 0:
            raise ValueError("gap_atr must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(14, min_periods=14).mean()
        gap_strength = (data["open"] - previous_close) / atr.shift(1).replace(0.0, np.nan)
        shelf_top = data["open"].shift(2)
        shelf_floor = previous_close.shift(2)
        setup = (
            (gap_strength.shift(2) >= params.gap_atr)
            & (data["close"].shift(2) > shelf_top)
            & (data["low"].shift(1) <= shelf_top)
            & (data["low"].shift(1) > shelf_floor)
            & (data["close"].shift(1) > shelf_top)
            & (data["low"] <= shelf_top)
            & (data["low"] > shelf_floor)
            & (data["close"] > shelf_top)
            & atr.gt(0.0)
        ).fillna(False)
        return pd.DataFrame({
            "atr": atr,
            "gap_strength": gap_strength,
            "shelf_top": shelf_top,
            "shelf_floor": shelf_floor,
            "setup": setup,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapShelfParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        pending_entry = False
        active = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.trail_atr * atrs[i - 1]
                    armed = False
                    active = True
            if active:
                valid_bar = np.isfinite(highs[i]) and np.isfinite(lows[i]) and np.isfinite(closes[i])
                exit_now = not valid_bar or lows[i] <= stop or i - entry_bar + 1 >= 10
                if not exit_now:
                    armed = armed or highs[i] >= entry * 1.015
                    if armed:
                        candidate = entry
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            candidate = max(candidate, closes[i] - params.trail_atr * atrs[i])
                        stop = max(stop, candidate)
                        exit_now = closes[i] <= stop
                if exit_now:
                    active = False
                else:
                    signals[i] = 1
            elif setups[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                signals[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([0, 1]).all() and df["size"].gt(0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
