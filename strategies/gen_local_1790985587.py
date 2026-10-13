from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CentroidExhaustionParams:
    center_window: int = 12
    rank_window: int = 60
    tail_fraction: float = 0.15
    profit_target: float = 0.015
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CentroidExhaustionParams]):
    strategy_id = "gen_local_1790985587"

    @classmethod
    def params_type(cls) -> type[CentroidExhaustionParams]:
        return CentroidExhaustionParams

    def warmup_bars(self, params: CentroidExhaustionParams) -> int:
        return int(params.center_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: CentroidExhaustionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        window = int(params.center_window)
        rank_window = int(params.rank_window)
        out["candle_center"] = (data["high"] + data["low"] + 2.0 * data["close"]) / 4.0
        out["valid_volume"] = data["volume"].where(data["volume"] > 0.0)
        out["weighted_center"] = out["candle_center"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(window, min_periods=window).sum()
        out["weighted_center_sum"] = out["weighted_center"].rolling(window, min_periods=window).sum()
        out["centroid"] = out["weighted_center_sum"] / out["volume_sum"].replace(0.0, np.nan)
        out["daily_range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["range_scale"] = out["daily_range"].rolling(window, min_periods=window).mean()
        out["displacement"] = (data["close"] - out["centroid"]) / out["range_scale"].replace(0.0, np.nan)
        out["displacement_rank"] = out["displacement"].rolling(rank_window, min_periods=rank_window).rank(method="average", pct=True)
        out["average_volume"] = out["volume_sum"] / float(window)
        out["volume_ratio"] = out["valid_volume"] / out["average_volume"].replace(0.0, np.nan)
        out["previous_volume"] = out["valid_volume"].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["exhaustion"] = ((out["volume_ratio"] < 1.0) & (out["valid_volume"] < out["previous_volume"])).fillna(False)
        out["long_setup"] = ((out["displacement_rank"] <= params.tail_fraction) & (out["displacement"] < 0.0) & out["exhaustion"]).fillna(False)
        out["short_setup"] = ((out["displacement_rank"] >= 1.0 - params.tail_fraction) & (out["displacement"] > 0.0) & out["exhaustion"]).fillna(False)
        out["previous_long_setup"] = out["long_setup"].shift(1, fill_value=False)
        out["previous_short_setup"] = out["short_setup"].shift(1, fill_value=False)
        out["long_confirmed"] = out["long_setup"] & out["previous_long_setup"]
        out["short_confirmed"] = out["short_setup"] & out["previous_short_setup"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CentroidExhaustionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        pending_entry = False
        hold_limit = max(1, int(params.max_hold_bars))

        for i in range(len(data)):
            # A close decision becomes a position at the following bar's open.
            if pending_entry:
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    position = 0

            if position != 0:
                held_bars = i - entry_bar + 1
                gain = np.nan
                if np.isfinite(closes[i]) and closes[i] > 0.0:
                    gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held_bars >= hold_limit:
                    position = 0
                    entry_price = np.nan
                    decisions[i] = 0
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                if longs[i]:
                    position = 1
                    pending_entry = True
                elif shorts[i]:
                    position = -1
                    pending_entry = True

            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
