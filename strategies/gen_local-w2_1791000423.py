from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDiscountParams:
    entry_z: float = 1.0
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalDiscountParams]):
    strategy_id = "gen_local-w2_1791000423"

    @classmethod
    def params_type(cls) -> type[SeasonalDiscountParams]:
        return SeasonalDiscountParams

    def warmup_bars(self, params: SeasonalDiscountParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: SeasonalDiscountParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["dispersion"] = close.rolling(20, min_periods=20).std(ddof=0)
        denominator = out["dispersion"].where(out["dispersion"] > 0.0)
        out["z"] = ((close - out["ma"]) / denominator).replace([np.inf, -np.inf], np.nan)
        out["z_previous"] = out["z"].shift(1)
        previous_close = close.shift(1)
        components = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1)
        out["true_range"] = components.max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        day = data.index.day
        out["seasonal_window"] = (day >= 25) | (day <= 5)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalDiscountParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        previous_z = indicators["z_previous"].to_numpy(dtype=float)
        seasonal = indicators["seasonal_window"].to_numpy(dtype=bool)
        valid = (
            np.isfinite(close)
            & np.isfinite(atr)
            & (atr > 0.0)
            & np.isfinite(z)
            & np.isfinite(previous_z)
        )
        entries = valid & seasonal & (z <= -params.entry_z) & (z > previous_z)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = -1
        highest_close = np.nan
        stop = np.nan

        for i in range(len(data)):
            if holding:
                if not valid[i]:
                    holding = False
                    highest_close = np.nan
                    stop = np.nan
                    continue

                # The entry decision occurred on the preceding bar;
                # this is the first close observed after its next-bar fill.
                if i == entry_bar:
                    highest_close = close[i]
                    stop = highest_close - params.stop_atr * atr[i]
                else:
                    highest_close = max(highest_close, close[i])
                    stop = max(stop, highest_close - params.stop_atr * atr[i])

                held_sessions = i - entry_bar + 1
                if close[i] <= stop or held_sessions >= 8:
                    holding = False
                    highest_close = np.nan
                    stop = np.nan
                else:
                    signals[i] = 1
            elif entries[i]:
                holding = True
                entry_bar = i + 1
                highest_close = np.nan
                stop = np.nan
                signals[i] = 1

        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
