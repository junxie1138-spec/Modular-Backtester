from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_window: int = 8
    history: int = 160
    neighbors: int = 7
    hold_days: int = 4
    atr_period: int = 14
    min_edge: float = 0.002
    breakeven_pct: float = 0.0075
    trail_atr: float = 1.75


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791231934"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if not (
            params.shape_window >= 3
            and params.history >= params.neighbors >= 1
            and 3 <= params.hold_days <= 5
            and params.atr_period >= 2
            and params.min_edge > 0
            and params.breakeven_pct > 0
            and params.trail_atr > 0
        ):
            raise ValueError("Invalid analog or stop parameters")
        return max(
            params.atr_period + 1,
            params.history + 2 * params.shape_window + params.hold_days,
        )

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["day_return"] = close.pct_change(fill_method=None)
        out["window_return"] = close / close.shift(params.shape_window - 1) - 1
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - close.shift(1)).abs(),
                (data["low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()

        n, w = len(data), params.shape_window
        path = np.column_stack(
            [out["log_close"].shift(w - 1 - j).to_numpy() for j in range(w)]
        )
        degree = np.zeros((n, w), dtype=float)
        # An edge exists when its slope exceeds every intervening slope.
        for j in range(w - 1):
            highest_slope = np.full(n, -np.inf)
            for k in range(j + 1, w):
                slope = (path[:, k] - path[:, j]) / (k - j)
                visible = slope > highest_slope
                degree[:, j] += visible
                degree[:, k] += visible
                highest_slope = np.maximum(highest_slope, slope)
        degree /= w - 1
        degree[~np.isfinite(path).all(axis=1)] = np.nan
        for j in range(w):
            out[f"degree_{j}"] = degree[:, j]

        distances = np.full((n, params.history), np.inf)
        outcomes = np.zeros((n, params.history))
        prices = close.to_numpy()
        h = params.hold_days
        # Candidate outcomes finish before the current shape starts.
        for column, lag in enumerate(range(w + h, w + h + params.history)):
            if lag >= n:
                continue
            distances[lag:, column] = np.sqrt(
                np.mean((degree[lag:] - degree[:-lag]) ** 2, axis=1)
            )
            outcomes[lag:, column] = (
                prices[h : n - lag + h] / prices[: n - lag] - 1
            )
        valid = np.isfinite(distances) & np.isfinite(outcomes)
        distances = np.where(valid, distances, np.inf)
        chosen = np.argpartition(
            distances, params.neighbors - 1, axis=1
        )[:, : params.neighbors]
        nearest = np.take_along_axis(distances, chosen, axis=1)
        returns = np.take_along_axis(outcomes, chosen, axis=1)
        weights = np.where(np.isfinite(nearest), 1 / (0.03 + nearest), 0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        total = weights.sum(axis=1)
        denominator = np.where(total > 0, total, 1.0)
        out["forecast"] = (weights * returns).sum(axis=1) / denominator
        out["mean_abs_outcome"] = (weights * np.abs(returns)).sum(axis=1) / denominator
        out["win_probability"] = (weights * (returns > 0)).sum(axis=1) / denominator
        out["graph_distance"] = np.where(
            total > 0,
            (weights * np.where(np.isfinite(nearest), nearest, 0.0)).sum(axis=1)
            / denominator,
            np.nan,
        )
        out["conviction"] = (
            out["forecast"] / out["mean_abs_outcome"].replace(0.0, np.nan)
        ).clip(0.0, 1.0).fillna(0.0)
        out["ready"] = (
            (valid.sum(axis=1) == params.history)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
            & out["day_return"].notna()
            & out["window_return"].notna()
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        enter = (
            indicators["ready"]
            & indicators["window_return"].lt(0)
            & indicators["day_return"].gt(0)
            & indicators["forecast"].ge(params.min_edge)
            & indicators["win_probability"].ge(0.55)
            & indicators["graph_distance"].le(0.20)
        ).fillna(False).to_numpy()
        strength = (
            0.20 + 0.75 * indicators["conviction"]
            * (1.0 - indicators["graph_distance"].clip(0.0, 1.0))
        ).fillna(0.20).clip(0.20, 0.95).to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        n = len(data)
        raw = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.20)
        active = pending = armed = False
        entry = peak = stop = 0.0
        allocation = 0.20
        held = 0

        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                    held = 0
                    active = True
            if active:
                held += 1
                sizes[i] = allocation
                # Existing stops can be breached intraday; fills remain next open.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                if np.isfinite(highs[i]):
                    peak = max(peak, highs[i])
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                # Newly raised stops are checked against the close, not earlier lows.
                breached = breached or (
                    np.isfinite(closes[i]) and closes[i] <= stop
                )
                if breached or held >= params.hold_days:
                    active = False
                else:
                    raw[i] = 1
            elif enter[i]:
                allocation = float(strength[i])
                sizes[i] = allocation
                raw[i] = 1
                pending = True

        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), AnalogParams()
    index = pd.date_range("2020-01-01", periods=320)
    t = np.arange(len(index))
    close = 100.0 * np.exp(0.0003 * t + 0.02 * np.sin(t / 3.0))
    data = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1000000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:270], params)
    pd.testing.assert_frame_equal(full.iloc[:270], prefix)
    signals = strategy.generate_signals(data, full, ctx, params).data
    partial = strategy.generate_signals(data.iloc[:270], prefix, ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:270], partial)
    assert signals["signal"].isin([0, 1]).all()
    assert signals["size"].between(0.20, 0.95).all()
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()

    sample = data.iloc[:5].copy()
    sample["open"] = [100.0, 100.0, 100.5, 103.0, 102.0]
    sample["high"] = [100.5, 101.0, 104.0, 103.5, 103.0]
    sample["low"] = [99.5, 99.9, 100.4, 102.0, 101.5]
    sample["close"] = [100.0, 100.5, 103.0, 102.5, 102.0]
    forced = pd.DataFrame(
        {"ready": [True, False, False, False, False],
         "window_return": -0.02, "day_return": 0.01, "forecast": 0.01,
         "win_probability": 1.0, "graph_distance": 0.0,
         "conviction": 1.0, "atr": 1.0},
        index=sample.index,
    )
    result = strategy.generate_signals(sample, forced, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result.loc[result["signal"].eq(1), "size"].eq(0.95).all()
    sample.loc[sample.index[2], "low"] = 99.9
    result = strategy.generate_signals(sample, forced, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
