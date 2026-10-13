from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ThursdayDrawdownParams:
    peak_lookback: int = 20
    percentile_lookback: int = 252
    drawdown_quantile: float = 0.75
    atr_lookback: int = 14
    atr_multiple: float = 2.0
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[ThursdayDrawdownParams]):
    strategy_id = "gen_local-w2_1790977116"

    @classmethod
    def params_type(cls) -> type[ThursdayDrawdownParams]:
        return ThursdayDrawdownParams

    def warmup_bars(self, params: ThursdayDrawdownParams) -> int:
        return int(max(params.peak_lookback + params.percentile_lookback + 1,
                       params.atr_lookback + 1))

    def indicators(self, data: pd.DataFrame, params: ThursdayDrawdownParams) -> pd.DataFrame:
        if params.peak_lookback < 2 or params.percentile_lookback < 126:
            raise ValueError("Lookbacks must be at least 2 and 126, respectively")
        if not 0.0 < params.drawdown_quantile < 1.0:
            raise ValueError("drawdown_quantile must be between zero and one")
        if params.atr_lookback < 1 or params.atr_multiple <= 0.0:
            raise ValueError("ATR settings must be positive")
        if not 3 <= params.holding_days <= 5:
            raise ValueError("holding_days must be between three and five")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_lookback,
                                    min_periods=params.peak_lookback).max()
        out["drawdown"] = (1.0 - close / out["peak"].where(out["peak"] > 0.0)).clip(lower=0.0)
        out["thursday"] = data.index.dayofweek == 3
        out["drawdown_threshold"] = (
            out["drawdown"].where(out["thursday"]).shift(1)
            .rolling(params.percentile_lookback, min_periods=20)
            .quantile(params.drawdown_quantile)
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_lookback, min_periods=params.atr_lookback
        ).mean()
        out["recovery"] = close.diff() > 0.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ThursdayDrawdownParams) -> SignalFrame:
        ready = indicators[["drawdown", "drawdown_threshold", "atr"]].notna().all(axis=1)
        entries = (
            ready & indicators["thursday"] & indicators["recovery"]
            & (indicators["drawdown"] > 0.0)
            & (indicators["drawdown"] >= indicators["drawdown_threshold"])
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]):
                    active = False
                else:
                    if age == 1:
                        high_water = max(opens[i], closes[i]) if np.isfinite(opens[i]) else closes[i]
                    else:
                        high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_multiple * atrs[i])
                    if closes[i] <= stop or age >= params.holding_days:
                        active = False
            elif entries[i]:
                active = True
                high_water = np.nan
                stop = -np.inf
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
