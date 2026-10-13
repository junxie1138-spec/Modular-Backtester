from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 15
    baseline_window: int = 5
    noise_window: int = 10
    innovation_threshold: float = 1.0
    minimum_strength: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790996122"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, max(3, params.slope_window) + max(1, params.baseline_window) + max(1, params.noise_window) + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        n = max(3, int(params.slope_window))
        baseline = max(1, int(params.baseline_window))
        noise_window = max(1, int(params.noise_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        total = out["log_close"].rolling(n, min_periods=n).sum()
        mean = total / n
        centered_time = time - (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        weighted_total = (out["log_close"] * time).rolling(n, min_periods=n).sum()
        out["slope"] = (weighted_total - centered_time * total) / denominator
        squared_total = out["log_close"].pow(2).rolling(n, min_periods=n).sum()
        residual_sum = (squared_total - n * mean.pow(2) - denominator * out["slope"].pow(2)).clip(lower=0.0)
        out["residual_noise"] = np.sqrt(residual_sum / (n - 2))
        out["slope_error"] = (out["residual_noise"] / np.sqrt(denominator)).clip(lower=1e-8)
        out["expected_slope"] = out["slope"].shift(1).rolling(baseline, min_periods=baseline).mean()
        out["innovation_z"] = (out["slope"] - out["expected_slope"]) / out["slope_error"]
        out["trend_strength"] = out["slope"] / out["slope_error"]
        out["noise_reference"] = out["residual_noise"].shift(1).rolling(noise_window, min_periods=noise_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        threshold = max(0.0, float(params.innovation_threshold))
        strength = max(0.0, float(params.minimum_strength))
        valid = indicators.notna().all(axis=1)
        quiet = indicators["residual_noise"] < indicators["noise_reference"]
        long_entry = (
            valid & quiet
            & (data["close"] > indicators["ma200"])
            & (indicators["trend_strength"] > strength)
            & (indicators["innovation_z"] > threshold)
            & (indicators["innovation_z"].shift(1) <= threshold)
        )
        short_entry = (
            valid & quiet
            & (data["close"] < indicators["ma200"])
            & (indicators["trend_strength"] < -strength)
            & (indicators["innovation_z"] < -threshold)
            & (indicators["innovation_z"].shift(1) >= -threshold)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        hold = min(5, max(3, int(params.hold_bars)))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar >= hold:
                    position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
