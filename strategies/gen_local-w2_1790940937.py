from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalForecastParams:
    fractional_order: float = 0.65
    fractional_window: int = 24
    state_window: int = 32
    correlation_window: int = 20
    forecast_threshold: float = 0.25
    min_correlation: float = 0.15
    atr_window: int = 14
    stop_atr: float = 1.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[FractionalForecastParams]):
    strategy_id = "gen_local-w2_1790940937"

    @classmethod
    def params_type(cls):
        return FractionalForecastParams

    def warmup_bars(self, params: FractionalForecastParams) -> int:
        return int(max(
            max(2, params.fractional_window)
            + max(2, params.state_window)
            + max(3, params.correlation_window) + 1,
            max(2, params.atr_window) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalForecastParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        window = max(2, int(params.fractional_window))
        order = float(np.clip(params.fractional_order, 0.01, 0.99))
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, window):
            weight *= (lag - 1 - order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        state_window = max(2, int(params.state_window))
        out["state_mean"] = fractional.rolling(state_window, min_periods=state_window).mean()
        out["state_std"] = fractional.rolling(state_window, min_periods=state_window).std(ddof=0)
        out["state_z"] = (fractional - out["state_mean"]) / out["state_std"].replace(0.0, np.nan)
        out["fractional_lag"] = fractional.shift(1)
        correlation_window = max(3, int(params.correlation_window))
        out["autocorrelation"] = fractional.rolling(
            correlation_window, min_periods=correlation_window
        ).corr(out["fractional_lag"]).clip(-1.0, 1.0)
        out["forecast"] = out["autocorrelation"] * out["state_z"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        atr_window = max(2, int(params.atr_window))
        out["atr"] = out["true_range"].rolling(atr_window, min_periods=atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalForecastParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = (
            indicators[["forecast", "autocorrelation", "atr"]].notna().all(axis=1)
            & data["close"].notna()
            & indicators["atr"].gt(0.0)
        )
        setup = (
            valid
            & indicators["autocorrelation"].gt(max(0.0, params.min_correlation))
            & indicators["forecast"].gt(max(0.0, params.forecast_threshold))
        )
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        valid_values = valid.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        trailing_stop = np.nan
        holding_bars = min(2, max(1, int(params.holding_bars)))
        stop_multiple = max(0.01, float(params.stop_atr))
        for i in range(len(data)):
            if active:
                age += 1
                if not valid_values[i]:
                    active = False
                else:
                    # The first close after the entry decision belongs to the filled trade.
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    candidate = high_water - stop_multiple * atr[i]
                    trailing_stop = candidate if np.isnan(trailing_stop) else max(trailing_stop, candidate)
                    if close[i] <= trailing_stop or age >= holding_bars:
                        active = False
            elif confirmed[i]:
                active = True
                age = 0
                high_water = np.nan
                trailing_stop = np.nan
            signals[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
