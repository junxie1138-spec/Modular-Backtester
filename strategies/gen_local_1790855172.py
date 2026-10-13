from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 21
    noise_window: int = 20
    innovation_window: int = 3
    percentile_window: int = 126
    entry_percentile: float = 0.85
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790855172"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slope_window, params.noise_window) + params.innovation_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.slope_window, params.noise_window) < 2:
            raise ValueError("Slope and noise windows must be at least two.")
        if min(params.innovation_window, params.percentile_window, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not 0.0 < params.entry_percentile < 1.0 or params.profit_target <= 0.0:
            raise ValueError("Invalid percentile or profit target.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        out["return"] = out["log_close"].diff()
        out["noise"] = out["return"].rolling(params.noise_window).std().replace(0.0, np.nan)
        n = params.slope_window
        coordinates = np.arange(n, dtype=float) - (n - 1) / 2.0
        weights = coordinates / np.dot(coordinates, coordinates)
        slope = np.full(len(data), np.nan, dtype=float)
        if len(data) >= n:
            slope[n - 1:] = np.convolve(out["log_close"].to_numpy(), weights[::-1], mode="valid")
        out["latent_slope"] = slope
        out["latent_level"] = out["log_close"].rolling(n).mean() + out["latent_slope"] * ((n - 1) / 2.0)
        out["forecast"] = out["latent_level"].shift(1) + out["latent_slope"].shift(1)
        out["forecast_error"] = out["log_close"] - out["forecast"]
        out["slope_update"] = out["latent_slope"].diff()
        out["normalized_update"] = out["slope_update"] / out["noise"]
        out["innovation"] = out["normalized_update"].rolling(params.innovation_window).mean()
        out["threshold"] = out["innovation"].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        out["eligible"] = (
            out["innovation"].gt(out["threshold"])
            & out["innovation"].gt(0.0)
            & out["latent_slope"].gt(0.0)
            & out["forecast_error"].gt(0.0)
            & np.isfinite(out["innovation"])
            & np.isfinite(out["threshold"])
        ).fillna(False)
        out["entry_ready"] = out["eligible"] & ~out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    # Observe the entry open only after the prior close's decision.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        active = False
                        continue
                    entry_bar = i
                    entry_price = opens[i]
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.max_hold
                if target_hit or timed_out:
                    active = False
                    entry_bar = -1
                    continue
                desired[i] = 1
            elif entries[i]:
                active = True
                entry_bar = -1
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    sample = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    sample.loc[index[2], "close"] = 105.0
    entries = pd.DataFrame({"entry_ready": [True, False, False, False, True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(sample, entries, None, Params(max_hold=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0.0).all()
