from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    box_bars: int = 8
    baseline_bars: int = 40
    compression_max: float = 0.70
    pressure_min: float = 0.25
    displacement_max: float = 0.35
    volume_min: float = 1.40
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791310476"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return int(max(params.box_bars, params.baseline_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        if not (2 <= params.box_bars <= params.baseline_bars):
            raise ValueError("Require 2 <= box_bars <= baseline_bars")
        if not (0 < params.compression_max <= 1 and 0 < params.pressure_min <= 1):
            raise ValueError("Invalid compression or pressure threshold")
        if not (0 <= params.displacement_max <= 1 and params.volume_min > 0):
            raise ValueError("Invalid displacement or volume threshold")
        if not (params.profit_target > 0 and params.hold_bars >= 1):
            raise ValueError("Invalid profit target or holding period")

        n = params.box_bars
        m = params.baseline_bars
        x = pd.DataFrame(index=data.index)
        x["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        x["body_direction"] = np.sign(data["close"] - data["open"])
        x["signed_volume"] = x["body_direction"] * data["volume"]
        x["short_range"] = x["daily_range"].rolling(n, min_periods=n).mean().shift(1)
        x["baseline_range"] = x["daily_range"].rolling(m, min_periods=m).mean().shift(1)
        x["box_high"] = data["high"].rolling(n, min_periods=n).max().shift(1)
        x["box_low"] = data["low"].rolling(n, min_periods=n).min().shift(1)
        x["box_width"] = (x["box_high"] - x["box_low"]).replace(0, np.nan)
        x["volume_sum"] = data["volume"].rolling(n, min_periods=n).sum().shift(1)
        x["signed_volume_sum"] = x["signed_volume"].rolling(n, min_periods=n).sum().shift(1)
        x["compression"] = x["short_range"] / x["baseline_range"].replace(0, np.nan)
        x["pressure"] = x["signed_volume_sum"] / x["volume_sum"].replace(0, np.nan)
        x["displacement"] = (data["close"].shift(1) - data["close"].shift(n)).abs() / x["box_width"]
        x["volume_ratio"] = data["volume"] / (x["volume_sum"] / n).replace(0, np.nan)
        x["close_location"] = (data["close"] - data["low"]) / x["daily_range"].replace(0, np.nan)
        x["penetration"] = ((data["close"] - x["box_high"]).clip(lower=0) + (x["box_low"] - data["close"]).clip(lower=0)) / x["box_width"]

        required = ["compression", "pressure", "displacement", "volume_ratio", "close_location", "penetration"]
        x["ready"] = np.isfinite(x[required]).all(axis=1)
        loaded = x["ready"] & (x["compression"] <= params.compression_max) & (x["displacement"] <= params.displacement_max) & (x["volume_ratio"] >= params.volume_min)
        long_entry = loaded & (x["pressure"] >= params.pressure_min) & (data["close"] > x["box_high"]) & (x["close_location"] >= 0.70)
        short_entry = loaded & (x["pressure"] <= -params.pressure_min) & (data["close"] < x["box_low"]) & (x["close_location"] <= 0.30)
        x["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        x["strength"] = (x["pressure"].abs() / params.pressure_min) * (x["volume_ratio"] / params.volume_min) * (1 + x["penetration"])
        x["entry_size"] = 0.25 + 0.70 * ((x["strength"] - 1) / 3).clip(lower=0, upper=1)
        x = x.replace([np.inf, -np.inf], np.nan)
        x["entry_size"] = x["entry_size"].fillna(0.25)
        return x.fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorptionParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        decision_bar = -1
        entry_price = np.nan
        held_size = 0.25

        for i in range(count):
            if direction:
                # The prior close's decision fills at this bar's open.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                age = i - decision_bar
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = direction * (closes[i] / entry_price - 1) if valid_price and np.isfinite(closes[i]) else np.nan
                if not valid_price or gain >= params.profit_target or age >= params.hold_bars:
                    direction = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(entries[i])
                decision_bar = i
                held_size = float(np.clip(entry_sizes[i], 0.25, 0.95))
            signals[i] = direction
            sizes[i] = held_size if direction else 0.25

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction, target_close, hold, exit_bar in [(1, 105.0, 18, 3), (-1, 95.0, 18, 3), (1, 100.0, 3, 4)]:
        data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1000.0}, index=index)
        data.loc[index[2]:, "close"] = target_close
        indicators = pd.DataFrame({"entry": [direction] + [0] * 7, "entry_size": 0.65}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, AbsorptionParams(hold_bars=hold)).data
        assert result["signal"].iloc[0] == 0
        assert (result["signal"].iloc[1:exit_bar] == direction).all()
        assert (result["signal"].iloc[exit_bar:] == 0).all()
        assert np.allclose(result["size"].iloc[1:exit_bar], 0.65)
        assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0}, index=pd.date_range("2020-01-01", periods=60))
    derived = strategy.indicators(flat, AbsorptionParams())
    assert not derived.isna().any().any()
    assert (derived["entry"] == 0).all()


if __name__ == "__main__":
    _self_check()
