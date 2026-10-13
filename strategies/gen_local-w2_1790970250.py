from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 6
    volume_window: int = 3
    volume_threshold: float = 1.10
    persistence_threshold: float = 0.25
    coherence_threshold: float = 0.55

    def __post_init__(self):
        if not (2 <= self.lookback and 2 <= self.volume_window and self.lookback + self.volume_window + 1 <= 10):
            raise ValueError("Lookbacks must be at least two and total warmup at most ten.")
        if not (np.isfinite(self.volume_threshold) and self.volume_threshold > 1):
            raise ValueError("volume_threshold must be finite and greater than one.")
        if not (0 < self.persistence_threshold < 1 and 0 < self.coherence_threshold < 1):
            raise ValueError("Evidence thresholds must lie strictly between zero and one.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790970250"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.lookback + params.volume_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        out["volume_baseline"] = data["volume"].shift(1).rolling(params.volume_window, min_periods=params.volume_window).median()
        out["relative_volume"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["volume_confirmed"] = (out["relative_volume"] >= params.volume_threshold).astype(float)
        out["confirmed_return"] = out["return"] * out["volume_confirmed"]
        out["confirmed_pair"] = out["return"] * out["lag_return"] * out["volume_confirmed"]
        out["pair_magnitude"] = out["confirmed_pair"].abs()
        out["move_magnitude"] = out["confirmed_return"].abs()
        out["pair_sum"] = out["confirmed_pair"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["pair_magnitude_sum"] = out["pair_magnitude"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["move_sum"] = out["confirmed_return"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["move_magnitude_sum"] = out["move_magnitude"].rolling(params.lookback, min_periods=params.lookback).sum()
        out["persistence"] = out["pair_sum"] / out["pair_magnitude_sum"].where(out["pair_magnitude_sum"] > 0)
        out["coherence"] = out["move_sum"] / out["move_magnitude_sum"].where(out["move_magnitude_sum"] > 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        persistence = indicators["persistence"].fillna(0.0)
        coherence = indicators["coherence"].fillna(0.0)
        eligible = persistence >= params.persistence_threshold
        candidate = pd.Series(np.select(
            [eligible & (coherence >= params.coherence_threshold),
             eligible & (coherence <= -params.coherence_threshold)],
            [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        confirmed.iloc[:self.warmup_bars(params) - 1] = 0
        signals = np.zeros(len(data), dtype=int)
        state = 0
        for i, direction in enumerate(confirmed.to_numpy()):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                state = 0
            signals[i] = state
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo():
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=22)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert strategy.warmup_bars(params) == 10 and flat["signal"].eq(0).all()
    coherence = [0.0] * 10 + [0.8, 0.8, 0.0, -0.8, -0.8, -0.8, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0]
    evidence = pd.DataFrame({"persistence": 0.8, "coherence": coherence}, index=index)
    result = strategy.generate_signals(data, evidence, ctx, params).data
    assert result["signal"].tolist() == [0] * 12 + [1, 1, 1, 0, -1, -1, -1, 0, 1, 1]
    assert result["size"].gt(0).all()
    prefix = strategy.indicators(data.iloc[:15], params)
    assert prefix.equals(strategy.indicators(data, params).iloc[:15])


if __name__ == "__main__":
    demo()
