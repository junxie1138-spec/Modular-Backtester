from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    volume_window: int = 20
    participation_window: int = 5
    atr_window: int = 14
    regime_window: int = 10
    regime_threshold: float = 0.5
    participation_expansion: float = 0.15
    volume_confirmation: float = 1.25
    trail_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790924987"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200 + params.regime_window, params.volume_window + 1,
                       params.participation_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.volume_window, params.participation_window,
               params.atr_window, params.regime_window, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive")
        if params.trail_atr <= 0 or params.volume_confirmation <= 0:
            raise ValueError("ATR multiplier and volume confirmation must be positive")
        close = data["close"]
        previous = close.shift(1)
        volume = data["volume"].clip(lower=0)
        out = pd.DataFrame(index=data.index)
        out["change"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["regime_strength"] = (
            out["ma200"] - out["ma200"].shift(params.regime_window)
        ) / out["atr"].replace(0, np.nan)
        out["volume_baseline"] = volume.shift(1).rolling(params.volume_window).mean()
        out["up_volume"] = volume.where(out["change"] > 0, 0.0).where(out["change"].notna())
        out["fast_share"] = (
            out["up_volume"].rolling(params.participation_window).sum()
            / volume.rolling(params.participation_window).sum().replace(0, np.nan)
        )
        out["slow_share"] = (
            out["up_volume"].rolling(params.volume_window).sum()
            / volume.rolling(params.volume_window).sum().replace(0, np.nan)
        )
        out["confirmed_up"] = (
            (out["change"] > 0)
            & (volume >= params.volume_confirmation * out["volume_baseline"])
        )
        out["confirmation_count"] = out["confirmed_up"].astype(float).rolling(
            params.participation_window
        ).sum()
        out["recent_change"] = close.diff(params.participation_window)
        out["prior_pullback"] = close.shift(1) - close.shift(params.participation_window + 1)
        eligible = (close > out["ma200"]) & (out["atr"] > 0)
        rising = out["regime_strength"] >= params.regime_threshold
        flat = out["regime_strength"].between(0.0, params.regime_threshold, inclusive="left")
        continuation = (
            rising & (out["recent_change"] > 0)
            & (out["fast_share"] - out["slow_share"] >= params.participation_expansion)
            & (out["confirmation_count"] >= 2)
            & out["confirmed_up"]
        )
        absorption = (
            flat & (out["prior_pullback"] < 0)
            & out["confirmed_up"] & (close > data["high"].shift(1))
        )
        out["entry"] = (eligible & (continuation | absorption)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        highest = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                highest, stop, age = np.nan, -np.inf, 0
                continue
            if active:
                age += 1
                highest = close if np.isnan(highest) else max(highest, close)
                stop = max(stop, highest - params.trail_atr * atr)
                if close <= stop or age >= params.holding_bars:
                    active = False
                    highest, stop, age = np.nan, -np.inf, 0
                else:
                    signals[i] = 1
            elif entries[i]:
                active = True
                highest, stop, age = np.nan, -np.inf, 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([0, 1]).all() and df["size"].gt(0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
