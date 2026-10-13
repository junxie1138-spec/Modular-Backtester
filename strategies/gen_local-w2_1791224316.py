from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletAnalogParams:
    history: int = 126
    target_pct: float = 0.01


class GeneratedStrategy(BaseStrategy[WaveletAnalogParams]):
    strategy_id = "gen_local-w2_1791224316"

    @classmethod
    def params_type(cls) -> type[WaveletAnalogParams]:
        return WaveletAnalogParams

    @classmethod
    def warmup_bars(cls, params: WaveletAnalogParams) -> int:
        return params.history + 9

    def indicators(self, data: pd.DataFrame, params: WaveletAnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"]).diff().fillna(0.0)
        r = out["return"]
        out["wave2"] = (r - r.shift(1)).fillna(0.0)
        out["wave4"] = (r.rolling(2).sum() - r.shift(2).rolling(2).sum()).fillna(0.0)
        out["wave8"] = (r.rolling(4).sum() - r.shift(4).rolling(4).sum()).fillna(0.0)
        energy = out[["wave2", "wave4", "wave8"]].pow(2)
        total = energy.sum(axis=1).replace(0.0, np.nan)
        features = energy.div(total, axis=0).fillna(0.0)
        features["phase"] = np.sign(out["wave8"])
        x = features.to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        forecast = np.zeros(len(data), dtype=float)
        for t in range(self.warmup_bars(params), len(data)):
            candidates = np.arange(t - params.history, t - 1)
            distance = np.abs(x[candidates, :3] - x[t, :3]).sum(axis=1)
            distance += 0.4 * (x[candidates, 3] != x[t, 3])
            nearest = candidates[np.argsort(distance)[:12]]
            forecast[t] = np.median(close[nearest + 2] / close[nearest] - 1.0)
        out["analog_forecast"] = forecast
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletAnalogParams,
    ) -> SignalFrame:
        forecast = indicators["analog_forecast"].to_numpy(dtype=float)
        entries = np.where(forecast > 0.003, 1, np.where(forecast < -0.003, -1, 0))
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = 0.0
        for t in range(len(data)):
            if t and desired[t - 1] != position:
                position = desired[t - 1]
                if position:
                    entry_bar = t
                    entry_price = opens[t]
            if position:
                gain = position * (closes[t] / entry_price - 1.0)
                desired[t] = 0 if gain >= params.target_pct or t - entry_bar + 1 >= 2 else position
            else:
                desired[t] = entries[t]
        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
