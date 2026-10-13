from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReleaseParams:
    compression_ratio: float = 0.65
    stop_atr: float = 2.5

    def __post_init__(self):
        if not np.isfinite(self.compression_ratio) or not 0 < self.compression_ratio < 0.9:
            raise ValueError("compression_ratio must be between zero and 0.9")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")


class GeneratedStrategy(BaseStrategy[RangeReleaseParams]):
    strategy_id = "gen_local-w2_1791287404"

    @classmethod
    def params_type(cls):
        return RangeReleaseParams

    def warmup_bars(self, params):
        return 21

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["relative_range"] = out["range"] / data["close"].where(data["close"] > 0)
        out["fast_range"] = out["relative_range"].rolling(5, min_periods=5).mean()
        out["slow_range"] = out["relative_range"].rolling(20, min_periods=20).mean()
        out["ratio"] = out["fast_range"] / out["slow_range"].where(out["slow_range"] > 0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["close_position"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        out["floor"] = data["low"].rolling(5, min_periods=5).min()
        out["prior_floor"] = out["floor"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        raw = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        ratio = indicators["ratio"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        position = indicators["close_position"].to_numpy(dtype=float)
        floor = indicators["floor"].to_numpy(dtype=float)
        prior_floor = indicators["prior_floor"].to_numpy(dtype=float)
        armed_at = -100
        holding = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open. Freeze its ATR.
            if holding and entry_bar == i:
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    holding = False
                    continue
                stop = opening[i] - params.stop_atr * atr[i - 1]
            if holding:
                if not np.isfinite(close[i]) or close[i] < stop or i - entry_bar >= 17:
                    holding = False
                    armed_at = -100
                else:
                    raw[i] = 1
                continue
            valid = all(np.isfinite(v) for v in (
                close[i], opening[i], ratio[i], atr[i], position[i],
                floor[i], prior_floor[i],
            )) and close[i] > 0 and atr[i] > 0
            if not valid:
                armed_at = -100
                continue
            if ratio[i] <= params.compression_ratio:
                armed_at = i
            if (0 < i - armed_at <= 10 and 0.9 <= ratio[i] <= 1.3
                    and floor[i] > prior_floor[i] and position[i] >= 0.65
                    and close[i] > opening[i]):
                holding = True
                entry_bar = i + 1
                armed_at = -100
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0,
        "close": 101.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "ratio": 1.0, "atr": 2.0, "close_position": 0.75,
        "floor": 99.0, "prior_floor": 98.0,
    }, index=index)
    ind.iloc[0, ind.columns.get_loc("ratio")] = 0.6
    strategy = GeneratedStrategy()
    params = RangeReleaseParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0] + [1] * 18 + [0] * 5
    data.loc[index[3], "close"] = 110.0
    data.loc[index[4], "close"] = 96.0
    data.loc[index[5], "close"] = 94.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:7].tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
