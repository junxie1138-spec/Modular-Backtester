from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    rank_window: int = 60
    stop_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = "gen_local_1790934138"

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return max(params.rank_window + 3, 15)

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        if params.rank_window < 5 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("rank_window must be >= 5 and stop_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["ceiling"] = data["high"].rolling(3, min_periods=3).max()
        out["floor"] = data["low"].rolling(3, min_periods=3).min()
        out["compression"] = (out["ceiling"] - out["floor"]) / data["close"].where(data["close"] > 0)
        out["throughput"] = data["volume"].clip(lower=0).rolling(3, min_periods=3).sum() / out["range"].rolling(3, min_periods=3).sum().replace(0, np.nan)
        out["compression_rank"] = out["compression"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["throughput_rank"] = out["throughput"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["loaded"] = (out["compression_rank"] <= 0.2) & (out["throughput_rank"] >= 0.8)
        out["prior_ceiling"] = out["ceiling"].shift(1)
        out["prior_floor"] = out["floor"].shift(1)
        out["ready"] = out["loaded"].shift(1).eq(True) & out["atr"].gt(0)
        out["entry"] = np.select(
            [out["ready"] & data["close"].gt(out["prior_ceiling"]),
             out["ready"] & data["close"].lt(out["prior_floor"])],
            [1, -1], default=0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CapacityParams,
    ) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry_atr = np.nan
        stop = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            if position:
                if age == 0:
                    # Previous close's decision fills here; freeze its ATR distance.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                invalid = not np.isfinite(opens[i]) or not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = position * (closes[i] - stop) < 0
                if invalid or stopped or age >= 2:
                    position = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                position = int(entries[i])
                age = 0
                entry_atr = atr[i]
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": [100.0, 98.0, 100.0, 100.0, 102.0, 100.0], "volume": 1000.0},
        index=index,
    )
    derived = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, -1, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = CapacityParams(stop_atr=1.0)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, derived, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert strategy.indicators(data, params).index.equals(data.index)
    assert strategy.warmup_bars(params) >= params.rank_window + 3
