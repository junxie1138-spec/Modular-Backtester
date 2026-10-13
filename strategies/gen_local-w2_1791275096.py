from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    profit_target: float = 0.02
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791275096"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return 22

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: x > 0)
        out["log_close"] = np.log(close)
        out["return"] = out["log_close"].diff()
        # Twelve-point least-squares slope, using fixed centered weights.
        slope = pd.Series(0.0, index=data.index)
        for lag in range(12):
            slope = slope + (5.5 - lag) * out["log_close"].shift(lag)
        out["slope"] = slope / 143.0
        out["predicted_slope"] = out["slope"].shift(1).rolling(5, min_periods=5).mean()
        out["slope_innovation"] = out["slope"] - out["predicted_slope"]
        out["return_volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        scale = (out["return_volatility"] / np.sqrt(12.0)).where(lambda x: x > 0)
        out["surprise"] = out["slope_innovation"] / scale
        out["channel_low"] = data["low"].rolling(20, min_periods=20).min()
        out["channel_high"] = data["high"].rolling(20, min_periods=20).max()
        width = (out["channel_high"] - out["channel_low"]).where(lambda x: x > 0)
        out["position"] = (close - out["channel_low"]) / width
        out["previous_surprise"] = out["surprise"].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["slope", "predicted_slope", "surprise", "position", "previous_surprise"]].notna().all(axis=1)
        long_entry = ready & (out["position"] < 0.4) & (out["predicted_slope"] < 0) & (out["surprise"] > 0.35) & (out["previous_surprise"] <= 0.35)
        short_entry = ready & (out["position"] > 0.6) & (out["predicted_slope"] > 0) & (out["surprise"] < -0.35) & (out["previous_surprise"] >= -0.35)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.max_hold < 1:
            raise ValueError("max_hold must be at least one bar")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # A decision from the preceding close fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    gain = side * (closes[i] / entry_price - 1.0)
                    elapsed = i - entry_bar + 1
                    if gain >= params.profit_target or elapsed >= params.max_hold:
                        side = 0
            elif candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"candidate": np.zeros(25, dtype=int)}, index=index)
    indicators.iloc[0, 0] = 1
    signals = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams()).data["signal"]
    assert signals.iloc[0] == 0
    assert (signals.iloc[1:11] == 1).all()
    assert (signals.iloc[11:] == 0).all()
    data.loc[index[1], "open"] = 110.0
    data.loc[index[1], "close"] = 113.0
    signals = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams()).data["signal"]
    assert signals.iloc[1] == 1 and signals.iloc[2] == 0
    indicators.iloc[0, 0] = -1
    data.loc[index[1], "close"] = 107.0
    signals = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams()).data["signal"]
    assert signals.iloc[1] == -1 and signals.iloc[2] == 0
    derived = GeneratedStrategy.indicators(data, StrategyParams())
    assert derived.index.equals(data.index)
    assert (derived["candidate"].iloc[:21] == 0).all()
