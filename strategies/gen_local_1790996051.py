from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeExhaustionParams:
    lookback: int = 20
    profit_target: float = 0.006


class GeneratedStrategy(BaseStrategy[RangeExhaustionParams]):
    strategy_id = "gen_local_1790996051"

    @classmethod
    def params_type(cls) -> type[RangeExhaustionParams]:
        return RangeExhaustionParams

    def warmup_bars(self, params: RangeExhaustionParams) -> int:
        return max(2, int(params.lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: RangeExhaustionParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["baseline_range"] = out["range"].rolling(window, min_periods=window).mean().shift(1)
        out["shock_ratio"] = out["range"] / out["baseline_range"]
        out["prior_shock_ratio"] = out["shock_ratio"].shift(1)
        out["contraction_ratio"] = out["range"] / out["range"].shift(1)
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out["prior_close_location"] = out["close_location"].shift(1)
        out["close_in_prior_range"] = (data["close"] - data["low"].shift(1)) / out["range"].shift(1)
        out["contained"] = ((data["high"] <= data["high"].shift(1)) & (data["low"] >= data["low"].shift(1))).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeExhaustionParams) -> SignalFrame:
        ready = indicators[["prior_shock_ratio", "contraction_ratio", "prior_close_location", "close_in_prior_range"]].notna().all(axis=1)
        exhaustion = ready & (indicators["prior_shock_ratio"] >= 1.8) & indicators["contraction_ratio"].between(0.35, 0.75) & indicators["contained"].eq(1)
        long_entry = exhaustion & (indicators["prior_close_location"] <= 0.2) & (indicators["close_in_prior_range"] <= 0.35)
        short_entry = exhaustion & (indicators["prior_close_location"] >= 0.8) & (indicators["close_in_prior_range"] >= 0.65)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0
        target = max(float(params.profit_target), 0.000001)

        for i in range(len(data)):
            if position != 0:
                # A decision from the preceding close fills at this open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held_bars += 1
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price and np.isfinite(closes[i]) else np.nan
                if not valid_price or (np.isfinite(gain) and gain >= target) or held_bars >= 2:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
                # No re-entry on an exit decision; zero reaches the next open.
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_price = np.nan
                held_bars = 0
            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
