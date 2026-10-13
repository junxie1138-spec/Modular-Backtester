from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBudgetParams:
    peak_window: int = 30
    trend_window: int = 60
    budget_window: int = 5
    compression_ratio: float = 0.65
    minimum_depth: float = 0.003
    maximum_depth: float = 0.04
    atr_window: int = 14
    breakeven_pct: float = 0.009
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownBudgetParams]):
    strategy_id = "gen_local_1791326742"

    @classmethod
    def params_type(cls) -> type[DrawdownBudgetParams]:
        return DrawdownBudgetParams

    def warmup_bars(self, params: DrawdownBudgetParams) -> int:
        return int(max(params.peak_window + 2 * params.budget_window, params.trend_window + params.budget_window, params.atr_window + 1) + 2)

    def indicators(self, data: pd.DataFrame, params: DrawdownBudgetParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close.div(out["peak"].where(out["peak"] > 0))
        out["recent_budget"] = out["depth"].rolling(params.budget_window, min_periods=params.budget_window).max()
        out["prior_budget"] = out["recent_budget"].shift(params.budget_window)
        out["budget_ratio"] = out["recent_budget"].div(out["prior_budget"].where(out["prior_budget"] > 0))
        out["trend"] = close.rolling(params.trend_window, min_periods=params.trend_window).mean()
        out["prior_trend"] = out["trend"].shift(params.budget_window)
        out["prior_depth"] = out["depth"].shift(1)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["setup"] = (
            (out["budget_ratio"] <= params.compression_ratio)
            & (out["depth"] >= params.minimum_depth)
            & (out["depth"] <= params.maximum_depth)
            & (out["depth"] < out["prior_depth"])
            & (close > out["previous_close"])
            & (close > out["trend"])
            & (out["trend"] > out["prior_trend"])
            & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownBudgetParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0:
                    continue
                # The preceding decision fills at this bar's open.
                entry = opens[i]
                stop = entry - params.initial_stop_atr * atrs[i - 1]
                active = True
                armed = False
                age = 0
            if active:
                age += 1
                valid = np.isfinite(closes[i]) and np.isfinite(lows[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
                # Only the previously established stop is tested against today's low.
                # An observed breach produces an exit at the following open.
                if not valid or lows[i] <= stop or age >= params.max_hold_bars:
                    active = False
                    continue
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atrs[i])
                decisions[i] = 1
            elif confirmed[i]:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    prices = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [100.5, 100.5, 101.2, 102.2, 102.0, 101.0, 101.0, 101.0],
        "low": [99.5, 99.5, 99.5, 100.2, 100.2, 99.5, 99.5, 99.5],
        "close": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    features = pd.DataFrame({"atr": 1.0, "confirmed": [False, True, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = DrawdownBudgetParams()
    result = strategy.generate_signals(prices, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert result.index.equals(prices.index) and (result["size"] > 0).all()
    derived = strategy.indicators(prices, params)
    assert not derived["confirmed"].any()
