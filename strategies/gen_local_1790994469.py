from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ThrustGapParams:
    rank_window: int = 63
    thrust_percentile: float = 0.85
    atr_window: int = 14
    trail_atr: float = 1.0
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ThrustGapParams]):
    strategy_id = "gen_local_1790994469"

    @classmethod
    def params_type(cls) -> type[ThrustGapParams]:
        return ThrustGapParams

    def warmup_bars(self, params: ThrustGapParams) -> int:
        return int(params.atr_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: ThrustGapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["body"] = data["close"] - data["open"]
        out["scaled_body"] = out["body"] / out["previous_atr"].where(out["previous_atr"] > 0)
        out["scaled_body"] = out["scaled_body"].replace([np.inf, -np.inf], np.nan)
        out["thrust_rank"] = out["scaled_body"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["gap_support"] = data["low"] - out["previous_high"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ThrustGapParams,
    ) -> SignalFrame:
        momentum = (indicators["thrust_rank"] >= params.thrust_percentile) & (indicators["body"] > 0)
        preserved_gap = indicators["gap_support"] > 0
        eligible = (
            momentum & preserved_gap
            & indicators["atr"].gt(0)
            & np.isfinite(indicators["atr"])
            & np.isfinite(data["close"])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        hold_limit = max(1, min(2, int(params.max_hold_bars)))

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                # The first active bar is the actual entry session after shifting.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= hold_limit:
                    active = False
                    continue
                signals[i] = 1
            elif eligible[i] and valid:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
