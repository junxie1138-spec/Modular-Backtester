from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 3
    baseline_window: int = 20
    atr_window: int = 14
    energy_share: float = 0.70
    burst_threshold: float = 1.20
    stop_atr: float = 1.80
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791315149"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(8 + params.energy_window + params.baseline_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if (min(params.energy_window, params.baseline_window, params.atr_window) < 1
                or not 3 <= params.hold_bars <= 5
                or not 0 < params.energy_share < 1
                or not np.isfinite(params.burst_threshold)
                or params.burst_threshold <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid wavelet parameters")
        out = pd.DataFrame(index=data.index)
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        out["mean2"] = out["log_price"].rolling(2, min_periods=2).mean()
        out["mean4"] = out["log_price"].rolling(4, min_periods=4).mean()
        # Causal, undecimated Haar details over 2, 4 and 8 closes.
        out["haar2"] = (out["log_price"] - out["log_price"].shift(1)) / np.sqrt(2.0)
        out["haar4"] = out["mean2"] - out["mean2"].shift(2)
        out["haar8"] = np.sqrt(2.0) * (out["mean4"] - out["mean4"].shift(4))
        out["noise_energy"] = out["haar2"].pow(2).rolling(params.energy_window).mean()
        out["trend_energy"] = (out["haar4"].pow(2) + out["haar8"].pow(2)).rolling(params.energy_window).mean()
        out["baseline"] = out["trend_energy"].shift(1).rolling(params.baseline_window).mean()
        total = out["noise_energy"] + out["trend_energy"]
        out["share"] = out["trend_energy"] / total.where(total > 0)
        out["burst"] = out["trend_energy"] / out["baseline"].where(out["baseline"] > 0)
        out["strength"] = (out["share"] * (out["burst"] / 2.0).clip(0, 1)).fillna(0).clip(0, 1)
        aligned = out["haar4"] * out["haar8"] > 0
        active = aligned & (out["share"] >= params.energy_share) & (out["burst"] >= params.burst_threshold)
        out["direction"] = np.sign(out["haar8"]).where(active, 0).fillna(0).astype(int)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        side, age = 0, 0
        entry, distance, position_size = np.nan, 0.0, 0.25
        for i in range(n):
            if side:
                if age == 0:
                    # The previous close ordered entry; this bar's open is the fill reference.
                    entry = opens[i]
                age += 1
                if (not np.isfinite(entry) or not np.isfinite(closes[i])
                        or side * (closes[i] - entry) < -distance
                        or age >= params.hold_bars):
                    side, age = 0, 0
                    entry = np.nan
                    continue
            elif (direction[i] and np.isfinite(closes[i])
                  and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(strength[i])):
                side = int(direction[i])
                age = 0
                distance = params.stop_atr * atr[i]
                position_size = 0.25 + 0.70 * float(np.clip(strength[i], 0, 1))
            signals[i] = side
            if side:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 110.0, "low": 90.0,
                             "close": [100, 100 + side * 6, 100 - side * 3,
                                       100 - side * 5, 100, 100, 100],
                             "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 2.0, "direction": [side, 0, 0, 0, 0, 0, 0],
                                    "strength": 0.8}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0, 0, 0]
        assert np.isclose(result["size"].iloc[1], 0.81)
        data["close"] = 100.0
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0, 0]
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=80))
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
