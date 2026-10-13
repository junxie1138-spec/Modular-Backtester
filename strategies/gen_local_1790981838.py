from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 20
    breakout_bars: int = 20
    min_far_visibility: float = 0.55
    profit_target: float = 0.03
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790981838"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if params.graph_bars < 3 or params.breakout_bars < 2:
            raise ValueError("Lookbacks must be at least three graph bars and two breakout bars")
        if not 0 < params.min_far_visibility <= 1:
            raise ValueError("min_far_visibility must be in (0, 1]")
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Profit target and holding period must be positive")
        return int(max(params.graph_bars + 1, params.breakout_bars + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0))
        degree = pd.Series(0.0, index=data.index)
        far_edges = pd.Series(0.0, index=data.index)
        nearest_min = pd.Series(np.inf, index=data.index)
        far_start = (params.graph_bars + 1) // 2
        # An older vertex is visible iff its backward slope is a strict record minimum.
        for lag in range(1, params.graph_bars):
            slope = (log_close - log_close.shift(lag)) / lag
            visible = slope.notna() & slope.lt(nearest_min)
            degree += visible.astype(float)
            if lag >= far_start:
                far_edges += visible.astype(float)
            nearest_min = pd.concat([nearest_min, slope], axis=1).min(axis=1)
        ready = log_close.rolling(params.graph_bars, min_periods=params.graph_bars).count().eq(params.graph_bars)
        out["log_close"] = log_close
        out["degree"] = degree.where(ready)
        out["far_visibility"] = (far_edges / (params.graph_bars - far_start)).where(ready)
        out["degree_gain"] = out["degree"].diff()
        high = data["high"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["prior_high"] = high.shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
        out["breakout"] = (close.gt(out["prior_high"]) & out["prior_high"].notna()).astype(int)
        out["entry"] = (
            out["breakout"].eq(1)
            & out["far_visibility"].ge(params.min_far_visibility)
            & out["degree_gain"].gt(0)
        ).astype(int)
        out["conviction_size"] = (0.25 + 0.70 * out["far_visibility"].fillna(0)).clip(0.25, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction_size"].fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if not valid_entry or target_hit or time_up:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entries[i] == 1 and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                held_size = conviction[i]
                signals[i] = 1
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 99.0, "close": [100.0, 100.0, 104.0, 104.0, 104.0, 104.0], "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0], "conviction_size": 0.60}, index=index)
    result = strategy.generate_signals(data, ind, None, VisibilityParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert np.allclose(result.loc[result["signal"].eq(1), "size"], 0.60)
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, VisibilityParams(max_hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    index = pd.date_range("2020-01-01", periods=40)
    close = 100.0 * np.exp(np.arange(40, dtype=float) ** 2 / 10000.0)
    data = pd.DataFrame({"open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, VisibilityParams())
    prefix = strategy.indicators(data.iloc[:30], VisibilityParams())
    assert np.isclose(full["far_visibility"].iloc[-1], 1.0)
    pd.testing.assert_frame_equal(full.iloc[:30], prefix)
