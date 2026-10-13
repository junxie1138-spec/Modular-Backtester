from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CongestionParams:
    compression_bars: int = 6
    baseline_bars: int = 30
    compression_ratio: float = 0.45
    volume_multiple: float = 1.3
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CongestionParams]):
    strategy_id = "gen_local_1791278388"

    @classmethod
    def params_type(cls):
        return CongestionParams

    def warmup_bars(self, params):
        return max(200, params.baseline_bars + params.compression_bars + 1)

    def indicators(self, data, params):
        if not (1 <= params.compression_bars < params.baseline_bars):
            raise ValueError("Require 1 <= compression_bars < baseline_bars")
        if not (0 < params.compression_ratio < 1 and params.volume_multiple > 0
                and params.profit_target > 0 and params.hold_bars >= 1):
            raise ValueError("Invalid thresholds or holding period")
        p = params
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["ceiling"] = data["high"].rolling(p.compression_bars).max().shift(1)
        out["floor"] = data["low"].rolling(p.compression_bars).min().shift(1)
        out["midpoint"] = (out["ceiling"] + out["floor"]) / 2
        out["width"] = out["ceiling"] - out["floor"]
        out["baseline_width"] = (
            data["high"].rolling(p.baseline_bars).max()
            - data["low"].rolling(p.baseline_bars).min()
        ).shift(1)
        out["compression"] = out["width"] / out["baseline_width"].replace(0, np.nan)
        out["jam_volume"] = data["volume"].rolling(p.compression_bars).mean().shift(1)
        out["baseline_volume"] = data["volume"].rolling(p.baseline_bars).mean().shift(p.compression_bars + 1)
        out["volume_ratio"] = data["volume"] / out["baseline_volume"].replace(0, np.nan)
        out["mean_range"] = (data["high"] - data["low"]).rolling(p.compression_bars).mean().shift(1)
        out["body"] = data["close"] - data["open"]
        out["previous_close"] = data["close"].shift(1)
        out["entry"] = (
            (data["close"] > out["ma200"])
            & (out["width"] > 0)
            & (out["compression"] <= p.compression_ratio)
            & (out["baseline_volume"] > 0)
            & (out["jam_volume"] >= 1.05 * out["baseline_volume"])
            & (out["volume_ratio"] >= p.volume_multiple)
            & (out["previous_close"] <= out["midpoint"])
            & (data["close"] > out["midpoint"])
            & (data["close"] <= out["ceiling"])
            & (out["mean_range"] > 0)
            & (out["body"] >= 0.6 * out["mean_range"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or expired:
                    active = False
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entry[i]:
                active = True
                entry_price = np.nan
                entry_bar = i + 1
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [90., 100., 100., 100., 100., 100., 100.],
        "high": [101.] * 7, "low": [89.] * 7,
        "close": [90., 100., 104., 100., 100., 100., 100.],
        "volume": [100.] * 7,
    }, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, True, False, False, False]}, index=index)
    params = CongestionParams(hold_bars=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    params.hold_bars = 4
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 1]
    assert not strategy.indicators(data, params)["entry"].any()
