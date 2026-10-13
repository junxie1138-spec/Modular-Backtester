from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeFloorParams:
    recent_bars: int = 3
    baseline_bars: int = 20
    shift_threshold: float = 0.75
    profit_target: float = 0.008
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[RangeFloorParams]):
    strategy_id = "gen_local_1791339056"

    @classmethod
    def params_type(cls):
        return RangeFloorParams

    def warmup_bars(self, params: RangeFloorParams) -> int:
        return max(200, params.baseline_bars + params.recent_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RangeFloorParams) -> pd.DataFrame:
        if params.recent_bars < 1 or params.baseline_bars < 2:
            raise ValueError("Range windows must be positive; baseline requires two bars.")
        if params.shift_threshold <= 0 or params.profit_target <= 0:
            raise ValueError("Changepoint threshold and profit target must be positive.")
        if params.max_hold not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")
        out = pd.DataFrame(index=data.index)
        width = (data["high"] - data["low"]).clip(lower=0)
        out["range_fraction"] = width / data["close"].where(data["close"] > 0)
        historical = out["range_fraction"].shift(params.recent_bars)
        out["baseline_mean"] = historical.rolling(params.baseline_bars).mean()
        out["baseline_std"] = historical.rolling(params.baseline_bars).std()
        out["range_floor"] = out["range_fraction"].rolling(params.recent_bars).min()
        out["floor_z"] = (
            (out["range_floor"] - out["baseline_mean"])
            / out["baseline_std"].where(out["baseline_std"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["body_pressure"] = (
            (data["close"] - data["open"]) / width.where(width > 0)
        )
        out["pressure_sum"] = out["body_pressure"].rolling(params.recent_bars).sum()
        out["ma200"] = data["close"].rolling(200).mean()
        out["changepoint"] = (
            (out["floor_z"] > params.shift_threshold)
            & (out["floor_z"].shift(1) <= params.shift_threshold)
        ).fillna(False)
        long_entry = (
            out["changepoint"] & (out["pressure_sum"] > 0)
            & (data["close"] > out["ma200"])
        )
        short_entry = (
            out["changepoint"] & (out["pressure_sum"] < 0)
            & (data["close"] < out["ma200"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeFloorParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The entry decision was yesterday; observe today's open now.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                gain = side * (closes[i] / entry_price - 1.0) if valid else 0.0
                # Close-observed target: the shifted flat signal exits next open.
                if not valid or gain >= params.profit_target or age >= params.max_hold:
                    side = 0
                    age = 0
                    entry_price = np.nan
                raw[i] = side
            elif entries[i] and np.isfinite(closes[i]):
                side = int(entries[i])
                age = 0
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeFloorParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 1, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 1, 1]
    assert (signals["size"] > 0).all()
    params.max_hold = 1
    entries["entry"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
