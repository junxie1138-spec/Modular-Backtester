from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSlopeParams:
    slope_window: int = 20
    history_window: int = 252
    entry_threshold: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SeasonalSlopeParams]):
    strategy_id = "gen_local_1791238207"

    @classmethod
    def params_type(cls):
        return SeasonalSlopeParams

    def warmup_bars(self, params):
        return int(max(200, params.history_window + params.slope_window + params.hold_bars + 2))

    def indicators(self, data, params):
        if params.slope_window < 2 or params.history_window < 126:
            raise ValueError("Require slope_window >= 2 and history_window >= 126")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("entry_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_open"] = np.log(data["open"].where(data["open"] > 0))
        out["return"] = out["log_close"].diff()
        out["latent_slope"] = out["return"].rolling(params.slope_window).mean().shift(1)
        out["volatility"] = out["return"].rolling(params.slope_window).std().shift(1)
        out["innovation"] = out["return"] - out["latent_slope"]
        out["z"] = out["innovation"] / out["volatility"].replace(0.0, np.nan)
        out["ma200"] = data["close"].rolling(200).mean()
        out["weekday"] = data.index.dayofweek
        lag = params.hold_bars + 1
        out["matured_z"] = out["z"].shift(lag)
        out["matured_weekday"] = out["weekday"].shift(lag)
        # Entry at the next open, exit exactly hold_bars opens later.
        out["realized_return"] = out["log_open"].diff(params.hold_bars)
        out["normalized_outcome"] = out["realized_return"] / out["volatility"].shift(lag).replace(0.0, np.nan)
        out["cross_product"] = out["matured_z"] * out["normalized_outcome"]
        out["squared_z"] = out["matured_z"].pow(2)
        valid = np.isfinite(out["cross_product"]) & np.isfinite(out["squared_z"])
        out["seasonal_beta"] = np.nan
        for weekday in range(5):
            observed = valid & out["matured_weekday"].eq(weekday)
            count = observed.astype(float).rolling(params.history_window).sum()
            numerator = out["cross_product"].where(observed, 0.0).rolling(params.history_window).sum()
            denominator = out["squared_z"].where(observed, 0.0).rolling(params.history_window).sum()
            out[f"weekday_count_{weekday}"] = count
            out[f"weekday_beta_{weekday}"] = (numerator / denominator.replace(0.0, np.nan)).where(count >= 20)
            selected = out["weekday"].eq(weekday)
            out.loc[selected, "seasonal_beta"] = out.loc[selected, f"weekday_beta_{weekday}"]
        out["score"] = (out["seasonal_beta"] * out["z"]).replace([np.inf, -np.inf], np.nan)
        out.loc[np.arange(len(out)) < self.warmup_bars(params) - 1, "score"] = np.nan
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        long_entry = indicators["score"].gt(params.entry_threshold) & data["close"].gt(indicators["ma200"])
        short_entry = indicators["score"].lt(-params.entry_threshold) & data["close"].lt(indicators["ma200"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        for i, candidate in enumerate(candidates):
            if position:
                age += 1
                if age >= params.hold_bars:
                    position = 0
                    age = 0
            elif candidate:
                position = int(candidate)
                age = 0
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    for direction in (1, -1):
        data = pd.DataFrame({"close": 100.0 + direction}, index=index)
        features = pd.DataFrame({"score": float(direction), "ma200": 100.0}, index=index)
        result = strategy.generate_signals(data, features, None, SeasonalSlopeParams()).data
        assert result["signal"].tolist() == [0, direction, direction, 0, direction, direction, 0, direction]
        features["score"] = [direction, -direction, -direction, 0, 0, 0, 0, 0]
        result = strategy.generate_signals(data, features, None, SeasonalSlopeParams()).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0, 0, 0]
    assert strategy.warmup_bars(SeasonalSlopeParams()) >= 276


if __name__ == "__main__":
    _self_check()
