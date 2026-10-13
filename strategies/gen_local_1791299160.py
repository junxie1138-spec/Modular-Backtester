from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureVolParams:
    vol_len: int = 20
    area_len: int = 12
    breakeven_pct: float = 0.03
    atr_mult: float = 2.5
    hold_bars: int = 21


class GeneratedStrategy(BaseStrategy[SignatureVolParams]):
    strategy_id = "gen_local_1791299160"

    @classmethod
    def params_type(cls):
        return SignatureVolParams

    def warmup_bars(self, params: SignatureVolParams) -> int:
        return params.vol_len + params.area_len + 64

    def indicators(self, data: pd.DataFrame, params: SignatureVolParams) -> pd.DataFrame:
        returns = np.log(data["close"] / data["close"].shift(1))
        sigma = returns.rolling(params.vol_len, min_periods=params.vol_len).std()
        ranges = np.log(data["high"] / data["low"])
        mean_range = ranges.rolling(params.vol_len, min_periods=params.vol_len).mean()
        price_step = (returns / sigma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        range_step = (ranges / mean_range.replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        local_area = 0.5 * (price_step.shift(1) * range_step - range_step.shift(1) * price_step)
        area = local_area.rolling(params.area_len, min_periods=params.area_len).mean()
        threshold = 0.8 * area.rolling(63, min_periods=63).std()
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        out = pd.DataFrame({"area": area, "threshold": threshold, "sigma": sigma, "atr": atr}, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureVolParams,
    ) -> SignalFrame:
        area = indicators["area"]
        threshold = indicators["threshold"]
        entry = (
            (area < -threshold)
            & (area.shift(1) >= -threshold.shift(1))
            & (threshold > 0)
        )
        strength = (area.abs() / (2 * threshold.replace(0, np.nan))).clip(0.25, 1.0)
        vol_scale = (0.12 / (np.sqrt(252.0) * indicators["sigma"].replace(0, np.nan))).clip(0.25, 1.0)
        sizes = (strength * vol_scale).fillna(0.25).clip(0.0625, 1.0)

        signals = np.zeros(len(data), dtype=int)
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        triggers = entry.fillna(False).to_numpy()
        active = False
        armed = False
        entry_price = stop = 0.0
        age = 0

        for i in range(self.warmup_bars(params), len(data)):
            if active:
                age += 1
                if lows[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    if highs[i] >= entry_price * (1 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price, highs[i] - params.atr_mult * atr[i])
            elif triggers[i] and atr[i] > 0:
                active = True
                armed = False
                entry_price = closes[i]
                stop = entry_price - params.atr_mult * atr[i]
                age = 0
            signals[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = sizes.shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
