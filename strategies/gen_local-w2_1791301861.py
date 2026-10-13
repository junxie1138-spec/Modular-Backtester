from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityDisplacementParams:
    ma_bars: int = 30
    energy_bars: int = 5
    vol_bars: int = 20
    atr_bars: int = 14
    max_hold_bars: int = 10
    z_entry: float = 1.2
    target_daily_vol: float = 0.012
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[VolatilityDisplacementParams]):
    strategy_id = "gen_local-w2_1791301861"

    @classmethod
    def params_type(cls) -> type[VolatilityDisplacementParams]:
        return VolatilityDisplacementParams

    @classmethod
    def warmup_bars(cls, params: VolatilityDisplacementParams) -> int:
        return max(params.ma_bars, params.energy_bars + 1, params.vol_bars + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: VolatilityDisplacementParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        deviation = close.rolling(params.ma_bars, min_periods=params.ma_bars).std()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)

        out = pd.DataFrame(index=data.index)
        out["z"] = ((close - mean) / deviation.replace(0.0, np.nan)).fillna(0.0)
        out["up_energy"] = returns.clip(lower=0.0).pow(2).rolling(params.energy_bars, min_periods=params.energy_bars).sum().fillna(0.0)
        out["down_energy"] = returns.clip(upper=0.0).pow(2).rolling(params.energy_bars, min_periods=params.energy_bars).sum().fillna(0.0)
        out["daily_vol"] = returns.rolling(params.vol_bars, min_periods=params.vol_bars).std().fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityDisplacementParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        state = 0
        entry_bar = 0
        high_water = 0.0
        low_water = 0.0
        stop = 0.0
        warmup = self.warmup_bars(params)

        for i in range(warmup, len(data)):
            price = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0.0:
                signal[i] = state
                continue

            if state == 1:
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.trail_atr * atr)
                if price <= stop or i - entry_bar >= params.max_hold_bars:
                    state = 0
            elif state == -1:
                low_water = min(low_water, price)
                stop = min(stop, low_water + params.trail_atr * atr)
                if price >= stop or i - entry_bar >= params.max_hold_bars:
                    state = 0
            elif indicators["daily_vol"].iloc[i] > 0.0:
                z = float(indicators["z"].iloc[i])
                up = float(indicators["up_energy"].iloc[i])
                down = float(indicators["down_energy"].iloc[i])
                if z >= params.z_entry and up > down:
                    state = 1
                    entry_bar = i
                    high_water = price
                    stop = price - params.trail_atr * atr
                elif z <= -params.z_entry and down > up:
                    state = -1
                    entry_bar = i
                    low_water = price
                    stop = price + params.trail_atr * atr
            signal[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (params.target_daily_vol / indicators["daily_vol"].replace(0.0, np.nan)).clip(lower=0.05, upper=0.95).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
