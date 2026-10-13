from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    atr_window: int = 14
    drawdown_window: int = 12
    shock_window: int = 5
    depth_atr: float = 2.5
    recovery_atr: float = 0.6
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791297534"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(200, params.atr_window, params.drawdown_window + params.shock_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1).fillna(data["close"])
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ma200 = data["close"].rolling(200, min_periods=200).mean()
        previous_high = data["high"].rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max().shift(1)
        drawdown = (previous_high - data["close"]) / atr.replace(0, np.nan)
        recent_drawdown = drawdown.rolling(
            params.shock_window, min_periods=params.shock_window
        ).max().shift(1)
        rebound = (data["close"] - data["close"].rolling(4, min_periods=4).min()) / atr.replace(0, np.nan)
        return pd.DataFrame({
            "atr": atr.fillna(0.0),
            "ma200": ma200.fillna(0.0),
            "drawdown": drawdown.fillna(0.0),
            "recent_drawdown": recent_drawdown.fillna(0.0),
            "rebound": rebound.fillna(0.0),
            "ready": ma200.notna() & atr.gt(0) & previous_high.notna(),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        atr = indicators["atr"]
        falling_atr = (atr < atr.shift(1)) & (atr.shift(1) < atr.shift(2))
        prior_drawdown = indicators["drawdown"].shift(1)
        common = indicators["ready"] & falling_atr & (
            indicators["recent_drawdown"] >= params.depth_atr
        )
        buy = (common & (data["close"] > indicators["ma200"])
               & (data["close"] > data["close"].shift(1))
               & (prior_drawdown - indicators["drawdown"] >= params.recovery_atr)).fillna(False).to_numpy()
        sell = (common & (data["close"] < indicators["ma200"])
                & (data["close"] < data["close"].shift(1))
                & (indicators["rebound"].shift(1) >= params.recovery_atr)).fillna(False).to_numpy()

        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr_values = atr.to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = target = age = 0
        entry = stop = 0.0
        armed = False

        for n in range(len(data)):
            if target != position:
                position = target
                age = 0
                armed = False
                if position:
                    entry = opens[n]
                    entry_atr = atr_values[n - 1]
                    stop = entry - position * params.trail_atr * entry_atr
            if position == 1:
                age += 1
                if highs[n] >= entry * (1 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, highs[n] - params.trail_atr * atr_values[n])
                if closes[n] <= stop or age >= params.max_hold:
                    target = 0
            elif position == -1:
                age += 1
                if lows[n] <= entry * (1 - params.breakeven_pct):
                    armed = True
                if armed:
                    stop = min(stop, entry, lows[n] + params.trail_atr * atr_values[n])
                if closes[n] >= stop or age >= params.max_hold:
                    target = 0
            elif buy[n]:
                target = 1
            elif sell[n]:
                target = -1
            desired[n] = target

        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
