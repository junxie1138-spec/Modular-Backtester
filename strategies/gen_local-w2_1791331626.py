from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownQueueParams:
    peak_window: int = 63
    memory_window: int = 10
    volatility_window: int = 14
    min_drawdown: float = 0.02
    trigger: float = 0.6
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownQueueParams]):
    strategy_id = "gen_local-w2_1791331626"

    @classmethod
    def params_type(cls):
        return DrawdownQueueParams

    def warmup_bars(self, params: DrawdownQueueParams) -> int:
        return int(max(params.peak_window + params.memory_window,
                       params.volatility_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownQueueParams) -> pd.DataFrame:
        if min(params.peak_window, params.memory_window, params.volatility_window,
               params.max_hold_bars) < 1:
            raise ValueError("Window lengths and holding limit must be positive.")
        if params.trigger <= 0 or params.profit_target <= 0 or params.min_drawdown < 0:
            raise ValueError("Invalid drawdown, trigger, or profit-target parameter.")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        previous_depth = out["depth"].shift(1)
        out["backlog"] = previous_depth.rolling(
            params.memory_window, min_periods=params.memory_window).mean()
        out["stress"] = previous_depth.rolling(
            params.memory_window, min_periods=params.memory_window).max()
        out["return"] = close.pct_change(fill_method=None)
        out["volatility"] = out["return"].abs().rolling(
            params.volatility_window, min_periods=params.volatility_window).mean()
        out["score"] = (out["backlog"] - out["depth"]) / out["volatility"].replace(0.0, np.nan)
        out["ready"] = out[["depth", "backlog", "stress", "score"]].notna().all(axis=1)
        out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["ready"] &= out["volatility"].gt(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownQueueParams) -> SignalFrame:
        score = indicators["score"]
        eligible = indicators["ready"].fillna(False) & indicators["stress"].ge(params.min_drawdown)
        previous_score = score.shift(1).fillna(0.0)
        buy = eligible & score.ge(params.trigger) & previous_score.lt(params.trigger)
        sell = eligible & score.le(-params.trigger) & previous_score.gt(-params.trigger)
        entries = np.select([buy, sell], [1, -1], default=0)
        conviction = (0.25 + 0.2 * score.abs() / params.trigger).clip(0.35, 0.95).fillna(0.35)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.35, dtype=float)
        side, entry_price, age, entry_size = 0, np.nan, 0, 0.35
        for i in range(len(data)):
            if side:
                # The previous close's decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                invalid = (not np.isfinite(entry_price) or entry_price <= 0
                           or not np.isfinite(closes[i]) or closes[i] <= 0)
                gain = side * (closes[i] / entry_price - 1.0) if not invalid else 0.0
                if invalid or gain >= params.profit_target or age >= params.max_hold_bars:
                    side, entry_price, age = 0, np.nan, 0
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_price, age = np.nan, 0
                entry_size = float(conviction.iloc[i])
            signals[i] = side
            sizes[i] = entry_size if side else 0.35
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=13)
    close = np.array([100, 100, 103, 103, 100, 100, 97, 97, 100, 100, 100, 100, 100], dtype=float)
    data = pd.DataFrame({"open": 100.0, "high": np.maximum(close, 100.0),
                         "low": np.minimum(close, 100.0), "close": close,
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 1, 1],
                               "stress": 0.04, "ready": True}, index=index)
    params = DrawdownQueueParams(profit_target=0.02, max_hold_bars=3)
    strategy = GeneratedStrategy()
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.25 + 0.2 / params.trigger)
    assert result["size"].gt(0).all()
