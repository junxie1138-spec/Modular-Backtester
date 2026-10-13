from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    relief_threshold: float = 0.4
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1790997538"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["positive_return"] = out["return"].clip(lower=0.0)
        out["negative_return"] = -out["return"].clip(upper=0.0)
        out["recent_upside"] = out["positive_return"].rolling(5, min_periods=5).sum()
        out["previous_upside"] = out["recent_upside"].shift(5)
        out["recent_downside"] = out["negative_return"].rolling(5, min_periods=5).sum()
        out["previous_downside"] = out["recent_downside"].shift(5)
        out["trend_return"] = out["return"].rolling(20, min_periods=20).sum()
        out["return_volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        scale = (out["return_volatility"] * np.sqrt(5.0)).replace(0.0, np.nan)
        out["relief"] = ((out["previous_downside"] - out["recent_downside"]) / scale).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        required = ["recent_upside", "previous_upside", "recent_downside", "previous_downside", "trend_return", "relief", "atr"]
        valid = pd.Series(np.isfinite(indicators[required].to_numpy(dtype=float)).all(axis=1), index=data.index)
        entry = (
            valid
            & (indicators["trend_return"] > 0.0)
            & (indicators["previous_downside"] > 0.0)
            & (indicators["recent_downside"] < indicators["previous_downside"])
            & (indicators["recent_upside"] >= 0.8 * indicators["previous_upside"])
            & (indicators["recent_upside"] > indicators["recent_downside"])
            & (indicators["relief"] >= params.relief_threshold)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=np.int64)
        active = False
        entry_bar = -1
        highest_close = 0.0
        stop = -np.inf
        multiplier = max(float(params.atr_multiple), 0.01)

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            usable = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                if not usable:
                    active = False
                    continue
                highest_close = max(highest_close, close)
                stop = max(stop, highest_close - multiplier * atr)
                if close <= stop or i - entry_bar >= 18:
                    active = False
                    continue
                decisions[i] = 1
            elif usable and entry[i]:
                active = True
                entry_bar = i
                highest_close = close
                stop = highest_close - multiplier * atr
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
