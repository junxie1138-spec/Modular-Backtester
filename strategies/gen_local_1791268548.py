from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionSlopeParams:
    slope_window: int = 8
    scale_window: int = 20
    shock_z: float = 1.5
    center_z_max: float = 0.6
    profit_target: float = 0.0075
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionSlopeParams]):
    strategy_id = "gen_local_1791268548"

    @classmethod
    def params_type(cls):
        return AuctionSlopeParams

    def warmup_bars(self, params):
        return int(max(200, params.slope_window + 2, params.scale_window + 2))

    def indicators(self, data, params):
        if params.slope_window < 2 or params.scale_window < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (params.shock_z > 0 and params.center_z_max > 0
                and params.profit_target > 0 and params.hold_bars in (1, 2)):
            raise ValueError("Invalid thresholds or holding horizon.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        midpoint = ((data["high"] + data["low"]) / 2).where(
            (data["high"] > 0) & (data["low"] > 0))
        out["log_close"] = np.log(close)
        out["log_midpoint"] = np.log(midpoint)
        out["close_return"] = out["log_close"].diff()
        out["midpoint_return"] = out["log_midpoint"].diff()
        out["latent_slope"] = out["midpoint_return"].rolling(
            params.slope_window, min_periods=params.slope_window).mean().shift(1)
        out["scale"] = out["close_return"].rolling(
            params.scale_window, min_periods=params.scale_window).std(ddof=0).shift(1)
        scale = out["scale"].where(out["scale"] > 1e-12)
        out["innovation"] = out["close_return"] - out["latent_slope"]
        out["center_innovation"] = out["midpoint_return"] - out["latent_slope"]
        out["innovation_z"] = out["innovation"] / scale
        out["center_z"] = out["center_innovation"] / scale
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        span = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = (close - data["low"]) / span
        valid = (
            np.isfinite(out["innovation_z"]) & np.isfinite(out["center_z"])
            & np.isfinite(out["ma200"]) & np.isfinite(out["close_location"])
            & out["center_z"].abs().le(params.center_z_max)
            & out["close_location"].between(0, 1)
        )
        long_entry = (
            valid & close.gt(out["ma200"])
            & out["innovation_z"].le(-params.shock_z)
            & out["close_location"].le(0.25)
        )
        short_entry = (
            valid & close.lt(out["ma200"])
            & out["innovation_z"].ge(params.shock_z)
            & out["close_location"].ge(0.75)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Reference the open of the delayed entry bar.
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1) if valid_entry else np.nan
                if (not valid_entry or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.hold_bars):
                    position = 0
                    entry_price = np.nan
                # Exit decisions cannot reenter on the same close.
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=5)
    bars = pd.DataFrame({"open": 100.0, "close": [100, 101, 100, 99, 100]}, index=index)
    impulses = pd.DataFrame({"entry": [1, 0, -1, 0, 0]}, index=index)
    result = strategy.generate_signals(bars, impulses, None, AuctionSlopeParams()).data
    assert result["signal"].tolist() == [0, 1, 0, -1, 0]
    bars["close"] = 100.0
    impulses["entry"] = [1, 0, 0, 0, 0]
    result = strategy.generate_signals(bars, impulses, None, AuctionSlopeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
