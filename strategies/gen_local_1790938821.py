from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownShockParams:
    peak_lookback: int = 20
    atr_lookback: int = 14
    volatility_lookback: int = 10
    min_depth_atr: float = 1.8
    contraction_ratio: float = 0.65
    stop_atr: float = 1.2
    target_volatility: float = 0.12
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownShockParams]):
    strategy_id = "gen_local_1790938821"

    @classmethod
    def params_type(cls) -> type[DrawdownShockParams]:
        return DrawdownShockParams

    def warmup_bars(self, params: DrawdownShockParams) -> int:
        return int(max(params.peak_lookback, params.atr_lookback + 1,
                       params.volatility_lookback + 1, 2))

    def indicators(self, data: pd.DataFrame, params: DrawdownShockParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["daily_range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_lookback,
                                                min_periods=params.atr_lookback).mean()
        out["peak"] = close.rolling(params.peak_lookback,
                                     min_periods=params.peak_lookback).max()
        out["depth_atr"] = (out["peak"] - close) / out["atr"].where(out["atr"] > 0)
        out["range_ratio"] = out["daily_range"] / out["daily_range"].shift(1).where(
            out["daily_range"].shift(1) > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["volatility"] = out["return"].rolling(
            params.volatility_lookback, min_periods=params.volatility_lookback
        ).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (params.target_volatility / out["volatility"].where(
            out["volatility"] > 0)).clip(lower=0.01, upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownShockParams) -> SignalFrame:
        if min(params.peak_lookback, params.atr_lookback, params.volatility_lookback) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not (0 < params.contraction_ratio < 1 and params.min_depth_atr > 0
                and params.stop_atr > 0 and params.target_volatility > 0
                and 0.01 <= params.max_size <= 1):
            raise ValueError("Invalid strategy parameters")

        ready = indicators[["depth_atr", "range_ratio", "atr", "target_size"]].notna().all(axis=1)
        agreement = (ready
                     & (indicators["depth_atr"] >= params.min_depth_atr)
                     & (indicators["range_ratio"] <= params.contraction_ratio))
        trigger = agreement & ~agreement.shift(1, fill_value=False)
        df = pd.DataFrame({"signal": 0, "size": 0.01}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target_size = indicators["target_size"].to_numpy(dtype=float)
        entries = trigger.to_numpy(dtype=bool)
        entry_bar = -1
        frozen_atr = 0.0
        stop_price = np.nan
        position_size = 0.01

        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                stop_hit = not np.isfinite(closes[i]) or closes[i] < stop_price
                expired = i - entry_bar + 1 >= 2
                if stop_hit or expired or not np.isfinite(stop_price):
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i] and i + 1 < len(data) and np.isfinite(opens[i + 1]) and opens[i + 1] > 0:
                entry_bar = i + 1
                frozen_atr = atr[i]
                position_size = target_size[i]
                signals[i] = 1
                sizes[i] = position_size

        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        assert df["signal"].isin([0, 1]).all() and (df["size"] > 0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
