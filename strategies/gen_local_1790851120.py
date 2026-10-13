from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShockParams:
    expansion_ratio: float = 1.8
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RangeShockParams]):
    strategy_id = "gen_local_1790851120"

    @classmethod
    def params_type(cls) -> type[RangeShockParams]:
        return RangeShockParams

    def warmup_bars(self, params: RangeShockParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: RangeShockParams) -> pd.DataFrame:
        if not np.isfinite(params.expansion_ratio) or params.expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must be finite and greater than one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["range_baseline"] = out["range"].rolling(20, min_periods=20).mean().shift(1)
        out["expansion"] = out["range"] / out["range_baseline"].where(out["range_baseline"] > 0.0)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_change"] = out["midpoint"].diff()
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        valid_prices = (data["low"] > 0.0) & (data["high"] >= data["low"])
        out["log_range"] = np.log((data["high"] / data["low"].where(data["low"] > 0.0)).where(valid_prices))
        out["range_variance"] = out["log_range"].pow(2) / (4.0 * np.log(2.0))
        out["annual_volatility"] = np.sqrt(out["range_variance"].rolling(20, min_periods=20).mean() * 252.0)
        out["target_size"] = (0.12 / out["annual_volatility"].where(out["annual_volatility"] > 0.0)).clip(0.10, 0.95).fillna(0.10)
        shock = out["expansion"].shift(1) >= params.expansion_ratio
        contraction = out["range"] < 0.75 * out["range"].shift(1)
        long_setup = shock & contraction & (out["midpoint_change"].shift(1) < 0.0) & (out["midpoint_change"] > 0.0) & (data["close"] > previous_close)
        short_setup = shock & contraction & (out["midpoint_change"].shift(1) > 0.0) & (out["midpoint_change"] < 0.0) & (data["close"] < previous_close)
        ready = np.isfinite(out["atr"]) & (out["atr"] > 0.0) & np.isfinite(out["annual_volatility"]) & (out["annual_volatility"] > 0.0)
        out["setup"] = np.select([long_setup & ready, short_setup & ready], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeShockParams) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.10, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].to_numpy(dtype=int)
        targets = indicators["target_size"].to_numpy(dtype=float)
        position = 0
        pending = 0
        entry_bar = 0
        entry_size = 0.10
        stop_price = np.nan
        for i in range(count):
            # A decision from yesterday fills at today's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = pending
                    entry_bar = i
                    stop_price = opens[i] - position * params.stop_atr * atr[i - 1]
                pending = 0
            if position:
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop_price) < 0.0
                expired = i - entry_bar + 1 >= 8
                if stopped or expired:
                    position = 0
                else:
                    decisions[i] = position
                sizes[i] = entry_size
            elif setups[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                pending = int(setups[i])
                entry_size = float(targets[i])
                decisions[i] = pending
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.10).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=60))
    prices.iloc[40] = [100.0, 100.0, 92.0, 93.0, 1000000.0]
    prices.iloc[41] = [93.0, 99.0, 95.0, 98.0, 1000000.0]
    prices.iloc[42] = [98.0, 99.0, 97.0, 98.0, 1000000.0]
    prices.iloc[43] = [98.0, 99.0, 70.0, 71.0, 1000000.0]
    strategy = GeneratedStrategy()
    params = RangeShockParams()
    derived = strategy.indicators(prices, params)
    result = strategy.generate_signals(prices, derived, None, params).data
    assert derived["setup"].iloc[41] == 1
    assert result["signal"].iloc[41] == 0
    assert result["signal"].iloc[42] == 1
    assert result["signal"].iloc[43] == 1
    assert result["signal"].iloc[44] == 0
    assert result["signal"].isin([-1, 0, 1]).all()
    assert np.isfinite(result["size"]).all() and (result["size"] > 0.0).all()
