from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReleaseParams:
    gap_window: int = 10
    atr_window: int = 14
    cancellation_limit: float = 0.25
    release_atr: float = 0.70
    acceptance: float = 0.65
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapReleaseParams]):
    strategy_id = "gen_local_1791000571"

    @classmethod
    def params_type(cls) -> type[GapReleaseParams]:
        return GapReleaseParams

    def warmup_bars(self, params: GapReleaseParams) -> int:
        return int(max(200, params.atr_window + 1, params.gap_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapReleaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["range"] = data["high"] - data["low"]
        out["high_distance"] = (data["high"] - out["prior_close"]).abs()
        out["low_distance"] = (data["low"] - out["prior_close"]).abs()
        out["true_range"] = out[["range", "high_distance", "low_distance"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_return"] = out["gap"] / out["prior_close"].where(out["prior_close"] > 0)
        out["prior_net_gap"] = out["gap_return"].rolling(params.gap_window, min_periods=params.gap_window).sum().shift(1)
        out["prior_gross_gap"] = out["gap_return"].abs().rolling(params.gap_window, min_periods=params.gap_window).sum().shift(1)
        out["cancellation"] = out["prior_net_gap"].abs() / out["prior_gross_gap"].where(out["prior_gross_gap"] > 0)
        out["gap_atr"] = out["gap"] / out["atr"].where(out["atr"] > 0)
        out["acceptance"] = (data["close"] - out["prior_close"]) / out["gap"].where(out["gap"] != 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapReleaseParams) -> SignalFrame:
        ready = indicators[["ma200", "atr", "cancellation", "gap_atr", "acceptance"]].notna().all(axis=1)
        ready &= (indicators["atr"] > 0) & (data["close"] > 0)
        release = ready & (indicators["cancellation"] <= params.cancellation_limit)
        release &= indicators["acceptance"] >= params.acceptance
        long_entry = release & (indicators["gap_atr"] >= params.release_atr) & (data["close"] > indicators["ma200"])
        short_entry = release & (indicators["gap_atr"] <= -params.release_atr) & (data["close"] < indicators["ma200"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan

        for i in range(len(data)):
            if position != 0:
                # The preceding close's entry decision fills at this open.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                stopped = np.isfinite(closes[i]) and position * (closes[i] - entry_price) <= -stop_distance
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid_entry or stopped or expired:
                    position = 0
                    signals[i] = 0
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                # Freeze volatility at the decision close; never trail the stop.
                stop_distance = params.stop_atr * atrs[i]
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
