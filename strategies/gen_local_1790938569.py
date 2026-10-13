from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    learning_bars: int = 252
    rank_bars: int = 126
    entry_percentile: float = 0.8
    holding_bars: int = 8
    profit_target: float = 0.018
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.learning_bars < 2 or self.rank_bars < 2 or self.holding_bars < 1:
            raise ValueError("Lookbacks must exceed one and holding_bars must be positive.")
        if not 0.0 < self.entry_percentile < 1.0 or self.profit_target <= 0.0:
            raise ValueError("Invalid percentile or profit target.")
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790938569"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params):
        return int(params.learning_bars + params.rank_bars)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["phase"] = 2.0 * np.pi * (data.index.day - 1) / data.index.days_in_month
        out["cos_phase"] = np.cos(out["phase"])
        out["sin_phase"] = np.sin(out["phase"])
        window = out["return"].rolling(params.learning_bars, min_periods=params.learning_bars)
        out["cos_coefficient"] = 2.0 * window.cov(out["cos_phase"], ddof=0)
        out["sin_coefficient"] = 2.0 * window.cov(out["sin_phase"], ddof=0)
        # ponytail: 1.4 calendar days per session; use an exchange calendar if holiday precision matters.
        out["forward_phase"] = out["phase"] + 2.0 * np.pi * 1.4 * (params.holding_bars + 1) / (2.0 * data.index.days_in_month)
        out["edge"] = out["cos_coefficient"] * np.cos(out["forward_phase"]) + out["sin_coefficient"] * np.sin(out["forward_phase"])
        out["strength"] = out["edge"].abs().rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        edges = indicators["edge"].to_numpy(dtype=float)
        ranks = indicators["strength"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction, entry_bar, entry_price, position_size = 0, -1, np.nan, params.min_size
        for i in range(len(data)):
            if direction:
                # Read the fill bar's open only after that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i])
                target_hit = valid_price and direction * (closes[i] / entry_price - 1.0) >= params.profit_target
                if target_hit or i - entry_bar + 1 >= params.holding_bars or not valid_price:
                    direction = 0
                    position_size = params.min_size
            elif np.isfinite(edges[i]) and np.isfinite(ranks[i]) and ranks[i] >= params.entry_percentile and edges[i] != 0.0:
                direction = 1 if edges[i] > 0.0 else -1
                entry_bar, entry_price = i + 1, np.nan
                conviction = np.clip((ranks[i] - params.entry_percentile) / (1.0 - params.entry_percentile), 0.0, 1.0)
                position_size = float(params.min_size + (params.max_size - params.min_size) * conviction)
            signals[i], sizes[i] = direction, position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    params = WaveParams(holding_bars=2)
    ind = pd.DataFrame({"edge": 0.01, "strength": 1.0}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:4].tolist() == [0, 1, 1, 0]
    assert result["size"].iloc[1] == params.max_size
    for side in (1, -1):
        data["close"] = [100.0, 100.0 + side * 2.0, 100.0, 100.0, 100.0, 100.0]
        ind["edge"] = side * 0.01
        result = strategy.generate_signals(data, ind, None, WaveParams(holding_bars=8)).data
        assert result["signal"].iloc[:3].tolist() == [0, side, 0]
    data["close"] = 100.0
    assert strategy.generate_signals(data, strategy.indicators(data, params), None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
