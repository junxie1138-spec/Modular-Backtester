from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityDominanceParams:
    dominance_threshold: float = 0.40
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[VolatilityDominanceParams]):
    strategy_id = "gen_local_1791258448"

    @classmethod
    def params_type(cls):
        return VolatilityDominanceParams

    def warmup_bars(self, params: VolatilityDominanceParams) -> int:
        return 25

    def indicators(self, data: pd.DataFrame, params: VolatilityDominanceParams) -> pd.DataFrame:
        if not np.isfinite(params.dominance_threshold) or not 0 < params.dominance_threshold < 1:
            raise ValueError("dominance_threshold must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["return"] = data["close"].div(previous_close.replace(0, np.nan)).sub(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["up_variance"] = out["return"].clip(lower=0).pow(2).rolling(8, min_periods=8).mean()
        out["down_variance"] = out["return"].clip(upper=0).pow(2).rolling(8, min_periods=8).mean()
        out["total_variance"] = out["up_variance"] + out["down_variance"]
        out["dominance"] = (out["up_variance"] - out["down_variance"]).div(out["total_variance"].replace(0, np.nan))
        out["short_volatility"] = out["return"].rolling(8, min_periods=8).std(ddof=0)
        out["background_volatility"] = out["return"].rolling(24, min_periods=24).std(ddof=0)
        out["drift"] = out["return"].rolling(8, min_periods=8).sum()
        out["trend_strength"] = out["drift"].div(out["background_volatility"].replace(0, np.nan) * np.sqrt(8.0))
        eligible = (
            out["atr"].gt(0)
            & out["short_volatility"].le(1.25 * out["background_volatility"])
        )
        long_entry = eligible & out["dominance"].ge(params.dominance_threshold) & out["trend_strength"].ge(1.0)
        short_entry = eligible & out["dominance"].le(-params.dominance_threshold) & out["trend_strength"].le(-1.0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityDominanceParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        side = 0
        entry_price = 0.0
        entry_bar = 0
        signed_stop = 0.0
        armed = False
        for i in range(len(data)):
            # A prior close decision fills at this session's open.
            previous_decision = decisions[i - 1] if i else 0
            if side and previous_decision == 0:
                side = 0
            if side == 0 and previous_decision != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    side = int(previous_decision)
                    entry_price = opens[i]
                    entry_bar = i
                    signed_stop = side * entry_price - 1.8 * atr[i - 1]
                    armed = False
            if side:
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    decisions[i] = 0
                    continue
                signed_close = side * closes[i]
                signed_entry = side * entry_price
                if signed_close - signed_entry >= 0.008 * entry_price:
                    armed = True
                if armed:
                    # Signed coordinates make both long and short stops ratchet upward.
                    signed_stop = max(signed_stop, signed_entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        signed_stop = max(signed_stop, signed_close - params.trail_atr * atr[i])
                # Stops are evaluated at the close; liquidation fills next open.
                stopped = signed_close <= signed_stop
                expired = i - entry_bar + 1 >= 4
                decisions[i] = 0 if stopped or expired else side
            elif i >= self.warmup_bars(params) - 1:
                if np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    decisions[i] = entries[i]
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
