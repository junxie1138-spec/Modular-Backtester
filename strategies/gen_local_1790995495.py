from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    response_threshold: float = 0.08
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790995495"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 550

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].rolling(8, min_periods=8).mean()
        out["volatility"] = out["return"].rolling(20, min_periods=20).std().shift(1)
        scale = out["volatility"].where(out["volatility"] > 1e-12)
        out["innovation"] = (out["slope"].diff() / scale).clip(-3.0, 3.0)
        out["weekday"] = data.index.dayofweek
        out["training_phase"] = out["weekday"].shift(1)
        out["training_x"] = out["innovation"].shift(1)
        out["training_y"] = out["return"]
        out["response"] = np.nan
        for day in range(5):
            prefix = "day_" + str(day) + "_"
            valid = (
                out["training_phase"].eq(day)
                & out["training_x"].notna()
                & out["training_y"].notna()
            )
            x = out["training_x"].where(valid, 0.0)
            y = out["training_y"].where(valid, 0.0)
            out[prefix + "count"] = valid.astype(float).rolling(504, min_periods=504).sum()
            out[prefix + "sum_x"] = x.rolling(504, min_periods=504).sum()
            out[prefix + "sum_y"] = y.rolling(504, min_periods=504).sum()
            out[prefix + "sum_xy"] = (x * y).rolling(504, min_periods=504).sum()
            out[prefix + "sum_xx"] = (x * x).rolling(504, min_periods=504).sum()
            count = out[prefix + "count"].where(out[prefix + "count"] >= 40)
            out[prefix + "covariance"] = (
                out[prefix + "sum_xy"]
                - out[prefix + "sum_x"] * out[prefix + "sum_y"] / count
            )
            out[prefix + "variance"] = (
                out[prefix + "sum_xx"] - out[prefix + "sum_x"] ** 2 / count
            )
            denominator = out[prefix + "variance"].where(out[prefix + "variance"] > 1e-12)
            out[prefix + "beta"] = out[prefix + "covariance"] / denominator
            phase = out["weekday"].eq(day)
            out.loc[phase, "response"] = out.loc[phase, prefix + "beta"]
        out["forecast"] = out["response"] * out["innovation"] / scale
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"]
        ready = (
            forecast.notna()
            & (np.arange(len(data)) >= self.warmup_bars(params))
        )
        threshold = max(float(params.response_threshold), 1e-12)
        candidates = np.select(
            [ready & forecast.gt(threshold), ready & forecast.lt(-threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        age = 0
        target = max(float(params.profit_target), 1e-12)
        for i in range(len(data)):
            if position:
                # The preceding close's entry decision fills at this open.
                if age == 0:
                    entry = opens[i]
                age += 1
                valid_entry = np.isfinite(entry) and entry > 0.0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                gain = position * (closes[i] / entry - 1.0) if valid_entry and valid_close else np.nan
                # Close-based exits execute at the following open.
                if not valid_entry or gain >= target or age >= 2:
                    position, entry, age = 0, np.nan, 0
                raw[i] = position
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position, entry, age = int(candidates[i]), np.nan, 0
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
