from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    diff_order: float = 0.45
    diff_window: int = 18
    baseline_window: int = 20
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791300491"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.diff_window + params.baseline_window + 2,
                   params.vol_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        weights = [1.0]
        for k in range(1, params.diff_window):
            weights.append(-weights[-1] * (params.diff_order - k + 1) / k)

        fractional_close = sum(
            weight * close.shift(k) for k, weight in enumerate(weights)
        )
        spread = fractional_close - fractional_close.rolling(
            params.baseline_window
        ).mean()

        previous_close = close.shift(1)
        true_range = pd.concat(
            [(data["high"] - data["low"]),
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        annual_vol = close.pct_change().rolling(params.vol_window).std(ddof=0) * np.sqrt(252)

        return pd.DataFrame({
            "fractional_close": fractional_close,
            "spread": spread,
            "atr": atr,
            "annual_vol": annual_vol,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        spread = indicators["spread"]
        confirmed = (
            spread.gt(0) & spread.shift(1).gt(0) & spread.shift(2).le(0)
        ).fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy()
        close = data["close"].to_numpy()
        opening = data["open"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        held = False
        entry_bar = -1
        stop = np.nan

        for i in range(len(data)):
            if held:
                if i == entry_bar:
                    stop = opening[i] - params.stop_atr * atr[i - 1]
                if close[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    held = False
                else:
                    raw[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                held = True
                entry_bar = i + 1
                raw[i] = 1

        volatility = indicators["annual_vol"].replace(0, np.nan)
        size = (params.target_vol / volatility).clip(lower=0.05, upper=1.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
