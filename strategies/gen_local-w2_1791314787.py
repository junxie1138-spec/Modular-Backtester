from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    regime_window: int = 5
    atr_window: int = 14
    tail_percentile: float = 0.80
    trail_atr: float = 1.5
    hold_bars: int = 2

    def __post_init__(self):
        if self.rank_window < 3 or self.regime_window < 1 or self.atr_window < 1:
            raise ValueError("Invalid lookback")
        if not 0.5 < self.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0 or self.hold_bars not in (1, 2):
            raise ValueError("Invalid trailing distance or holding horizon")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791314787"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window + 1, 2 * params.rank_window + params.regime_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        previous = out["previous_close"].where(out["previous_close"] > 0)
        out["return"] = data["close"] / previous - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["range_fraction"] = out["true_range"] / previous
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["volume_rank"] = data["volume"].where(data["volume"] > 0).rolling(params.rank_window).rank(pct=True)
        out["range_rank"] = out["range_fraction"].rolling(params.rank_window).rank(pct=True)
        out["return_rank"] = out["return"].rolling(params.rank_window).rank(pct=True)
        out["rank_gap"] = out["volume_rank"] - out["range_rank"]
        out["imbalance"] = out["rank_gap"].rolling(params.regime_window).mean()
        out["regime_rank"] = out["imbalance"].rolling(params.rank_window).rank(pct=True)
        out = out.replace([np.inf, -np.inf], np.nan)
        p = params.tail_percentile
        out["impulse"] = np.select([
            (out["return_rank"] >= p) & (out["return"] > 0),
            (out["return_rank"] <= 1.0 - p) & (out["return"] < 0),
        ], [1, -1], default=0).astype(int)
        out["regime"] = np.select([
            out["regime_rank"] <= 1.0 - p,
            out["regime_rank"] >= p,
        ], [1, -1], default=0).astype(int)
        out["entry"] = (out["impulse"] * out["regime"]).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = age = 0
        entry_atr = high_water = low_water = high_stop = low_stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = age = 0
                continue
            if position:
                if age == 0:
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        position = 0
                        continue
                    # The previous decision fills at today's open.
                    high_water = low_water = opening[i]
                    high_stop = opening[i] - params.trail_atr * entry_atr
                    low_stop = opening[i] + params.trail_atr * entry_atr
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                high_stop = max(high_stop, high_water - params.trail_atr * atr[i])
                low_stop = min(low_stop, low_water + params.trail_atr * atr[i])
                age += 1
                # Required high-water drawdown exit applies to both directions.
                # Shorts also have a mirrored adverse-move trailing stop.
                stopped = close[i] <= high_stop or (position == -1 and close[i] >= low_stop)
                if stopped or age >= params.hold_bars:
                    position = age = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_atr = atr[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6, "high": [103.0] * 6,
        "low": [97.0] * 6, "close": [100.0, 98.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data.loc[index[1], "close"] = 102.0
    ind.loc[index[0], "entry"] = -1
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
