from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_bars: int = 30
    z_bars: int = 50
    percentile_bars: int = 126
    entry_percentile: float = 0.80
    efficiency_bars: int = 20
    min_efficiency: float = 0.35
    vol_bars: int = 20
    target_vol: float = 0.15
    atr_bars: int = 14
    stop_atr: float = 3.0
    max_hold_bars: int = 22


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791226934"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(
            params.ma_bars + params.z_bars + params.percentile_bars + 2,
            params.efficiency_bars + 1,
            params.vol_bars + 1,
            params.atr_bars + 1,
        )

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        ma = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        distance = close / ma - 1.0
        distance_mean = distance.rolling(params.z_bars, min_periods=params.z_bars).mean()
        distance_std = distance.rolling(params.z_bars, min_periods=params.z_bars).std(ddof=0)
        z = (distance - distance_mean) / distance_std.replace(0.0, np.nan)
        threshold = z.abs().rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile).shift(1)

        travel = close.diff().abs().rolling(
            params.efficiency_bars, min_periods=params.efficiency_bars
        ).sum()
        efficiency = (close - close.shift(params.efficiency_bars)).abs() / travel.replace(0.0, np.nan)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        realized_vol = np.log(close / previous_close).rolling(
            params.vol_bars, min_periods=params.vol_bars
        ).std(ddof=0) * np.sqrt(252.0)
        size = (params.target_vol / realized_vol.replace(0.0, np.nan)).clip(
            lower=0.10, upper=1.0
        )
        ready = z.notna() & threshold.notna() & efficiency.notna() & atr.gt(0) & realized_vol.gt(0)

        return pd.DataFrame(
            {
                "z": z.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "threshold": threshold.replace([np.inf, -np.inf], np.nan).fillna(1e9),
                "efficiency": efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                "atr": atr.fillna(0.0),
                "size": size.replace([np.inf, -np.inf], np.nan).fillna(0.10),
                "ready": ready,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        z = indicators["z"]
        threshold = indicators["threshold"]
        qualified = indicators["ready"] & indicators["efficiency"].ge(params.min_efficiency)
        long_entry = qualified & z.gt(threshold) & z.shift(1).le(threshold.shift(1))
        short_entry = qualified & z.lt(-threshold) & z.shift(1).ge(-threshold.shift(1))

        positions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = low_water = stop = 0.0
        for i, (price, atr, go_long, go_short) in enumerate(
            zip(data["close"], indicators["atr"], long_entry, short_entry)
        ):
            exited = False
            if position == 1:
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.stop_atr * atr)
                exited = price <= stop or i - entry_bar >= params.max_hold_bars
            elif position == -1:
                low_water = min(low_water, price)
                stop = min(stop, low_water + params.stop_atr * atr)
                exited = price >= stop or i - entry_bar >= params.max_hold_bars
            if exited:
                position = 0
            elif position == 0 and go_long:
                position = 1
                entry_bar = i
                high_water = price
                stop = price - params.stop_atr * atr
            elif position == 0 and go_short:
                position = -1
                entry_bar = i
                low_water = price
                stop = price + params.stop_atr * atr
            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
