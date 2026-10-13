from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory_window: int = 3
    pressure_window: int = 2
    score_threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791323274"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if not 2 <= params.memory_window <= 4:
            raise ValueError("memory_window must be between 2 and 4")
        if not 1 <= params.pressure_window <= 4:
            raise ValueError("pressure_window must be between 1 and 4")
        if not np.isfinite(params.score_threshold) or params.score_threshold <= 0:
            raise ValueError("score_threshold must be finite and positive")
        return int(max(params.memory_window + 6, params.pressure_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        valid = data[["high", "low", "close"]].notna().all(axis=1) & out["previous_close"].notna()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1).where(valid)
        out["atr"] = out["true_range"].rolling(3, min_periods=3).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["range_shock"] = (out["true_range"] / out["prior_atr"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["lagged_shock"] = out["range_shock"].shift(1)
        out["shock_autocorrelation"] = out["range_shock"].rolling(params.memory_window, min_periods=params.memory_window).corr(out["lagged_shock"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["forecast"] = out["shock_autocorrelation"] * out["range_shock"]
        out["price_change"] = data["close"].diff(params.pressure_window)
        out["pressure"] = np.sign(out["price_change"])
        out["raw_score"] = out["forecast"] * out["pressure"]
        out["score"] = out["raw_score"].rolling(2, min_periods=2).mean().replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        scores = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        entries = np.where(scores > params.score_threshold, 1, np.where(scores < -params.score_threshold, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position != 0 and entry == -position:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [-0.2, -0.2, np.nan, 0.2, 0.2, 0.0, -0.2]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 1, 1]
    assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(Params(memory_window=4)) == 10
    flat = pd.concat([data] * 3, ignore_index=True)
    derived = GeneratedStrategy.indicators(flat, Params())
    assert derived["score"].isna().all()
    assert GeneratedStrategy.generate_signals(flat, derived, None, Params()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
