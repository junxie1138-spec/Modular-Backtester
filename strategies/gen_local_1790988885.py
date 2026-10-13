from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    roc_bars: int = 5
    acceleration_bars: int = 3
    channel_bars: int = 63
    percentile_bars: int = 126
    entry_quantile: float = 0.80
    atr_bars: int = 14
    trail_atr: float = 2.8
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = "gen_local_1790988885"

    @classmethod
    def params_type(cls) -> type[PropagationParams]:
        return PropagationParams

    def warmup_bars(self, params: PropagationParams) -> int:
        base = max(
            params.channel_bars,
            params.roc_bars + params.acceleration_bars + 1,
            params.atr_bars + 1,
        )
        return int(base + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: PropagationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["channel_low"] = close.rolling(params.channel_bars).min()
        out["channel_high"] = close.rolling(params.channel_bars).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["position"] = (
            (close - out["channel_low"])
            / out["channel_width"].where(out["channel_width"] > 0)
        ).clip(0.0, 1.0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff(params.acceleration_bars)
        out["headroom"] = 1.0 - out["position"]
        out["propagation"] = out["acceleration"] * out["headroom"]
        out["propagation_threshold"] = (
            out["propagation"].shift(1)
            .rolling(params.percentile_bars).quantile(params.entry_quantile)
        )
        out["position_median"] = (
            out["position"].shift(1)
            .rolling(params.percentile_bars).quantile(0.50)
        )
        out["position_upper"] = (
            out["position"].shift(1)
            .rolling(params.percentile_bars).quantile(0.85)
        )
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ], axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PropagationParams,
    ) -> SignalFrame:
        required = [
            "position", "position_median", "position_upper", "roc",
            "acceleration", "propagation", "propagation_threshold", "atr",
        ]
        ready = indicators[required].notna().all(axis=1)
        entries = (
            ready
            & (indicators["atr"] > 0)
            & (indicators["roc"] > 0)
            & (indicators["acceleration"] > 0)
            & (indicators["propagation"] > indicators["propagation_threshold"])
            & (indicators["position"] > indicators["position_median"])
            & (indicators["position"] < indicators["position_upper"])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            price = closes[i]
            atr = atrs[i]
            valid = np.isfinite(price) and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or price <= stop or age >= params.holding_bars:
                    active = False
                    high_water = -np.inf
                    stop = -np.inf
                    age = 0
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                active = True
                age = 0
                # The first in-trade close occurs on the next bar after filling.
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
