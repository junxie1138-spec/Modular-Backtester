from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionEntropyParams:
    complexity_drop: float = 0.08
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[PositionEntropyParams]):
    strategy_id = "gen_local-w2_1791000165"

    @classmethod
    def params_type(cls) -> type[PositionEntropyParams]:
        return PositionEntropyParams

    @staticmethod
    def warmup_bars(params: PositionEntropyParams) -> int:
        # Ten-bar positions, three-point patterns, twelve patterns,
        # and a three-bar entropy comparison require 26 observations.
        return 27

    @staticmethod
    def indicators(data: pd.DataFrame, params: PositionEntropyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range_high"] = data["high"].rolling(10, min_periods=10).max()
        out["range_low"] = data["low"].rolling(10, min_periods=10).min()
        out["range_width"] = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = ((data["close"] - out["range_low"]) / out["range_width"]).clip(0.0, 1.0)
        out["position_lag1"] = out["position"].shift(1)
        out["position_lag2"] = out["position"].shift(2)
        a = out["position_lag2"]
        b = out["position_lag1"]
        c = out["position"]
        valid = a.notna() & b.notna() & c.notna()
        # Earlier observations win ties, giving a deterministic total order.
        code = 4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)
        out["ordinal_code"] = code.astype(float).where(valid)
        probability_columns = []
        for pattern in (0, 1, 3, 4, 6, 7):
            column = f"probability_{pattern}"
            frequency = out["ordinal_code"].eq(pattern).astype(float).where(valid)
            out[column] = frequency.rolling(12, min_periods=12).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        entropy_terms = -probabilities * np.log(probabilities.clip(lower=1e-12))
        out["entropy"] = entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["entropy_lag3"] = out["entropy"].shift(3)
        out["entropy_contraction"] = out["entropy_lag3"] - out["entropy"]
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry_candidate"] = (
            out["entropy_contraction"].ge(params.complexity_drop)
            & out["position_lag1"].lt(0.35)
            & out["position"].ge(0.35)
            & out["position"].le(0.65)
            & data["close"].gt(out["previous_close"])
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PositionEntropyParams,
    ) -> SignalFrame:
        if params.trail_atr <= 0.0 or not 0.0 <= params.complexity_drop <= 1.0:
            raise ValueError("Invalid strategy parameters")
        count = len(data)
        raw = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry_candidate"].fillna(False).to_numpy(dtype=bool)
        pending_entry = False
        holding = False
        activated = False
        entry = 0.0
        stop = 0.0
        age = 0
        for i in range(count):
            if pending_entry:
                pending_entry = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    # The preceding close's decision fills at this open.
                    entry = opens[i]
                    stop = max(0.0, entry - params.trail_atr * atr[i - 1])
                    holding = True
                    activated = False
                    age = 0
            if holding:
                age += 1
                # Test the stop established before this bar; updates below
                # become effective on the next bar, avoiding intrabar ordering assumptions.
                stopped = np.isfinite(lows[i]) and lows[i] <= stop
                if stopped or age >= 4:
                    holding = False
                    raw[i] = 0
                    continue
                if np.isfinite(closes[i]):
                    if closes[i] >= entry * 1.008:
                        activated = True
                    if activated:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, closes[i] - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        holding = False
                        raw[i] = 0
                        continue
                raw[i] = 1
            elif candidates[i]:
                raw[i] = 1
                pending_entry = True
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
