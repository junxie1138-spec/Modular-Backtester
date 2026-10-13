from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    shape_bars: int = 12
    history_bars: int = 504
    holding_bars: int = 18
    atr_bars: int = 14
    bandwidth: float = 0.22
    edge_threshold: float = 0.25
    breakeven_pct: float = 0.015
    trail_atr: float = 3.0
    initial_stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791338042"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params):
        return int(params.holding_bars + 2 * params.shape_bars + params.history_bars + params.atr_bars + 1)

    def indicators(self, data, params):
        if min(params.shape_bars, params.history_bars, params.holding_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if min(params.bandwidth, params.edge_threshold, params.breakeven_pct, params.trail_atr, params.initial_stop_atr) <= 0:
            raise ValueError("Thresholds and stop distances must be positive.")
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        returns = np.log(close).diff()
        length = params.shape_bars
        variation = returns.abs().rolling(length).sum().replace(0, np.nan)
        # Antisymmetric second-order signature of (normalized time, log price).
        area = sum(
            ((j - (length - 1) / 2.0) / length) * returns.shift(length - 1 - j)
            for j in range(length)
        ) / variation
        endpoint = returns.rolling(length).sum() / variation
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars).mean()
        horizon = params.holding_bars
        # At t this label belongs to origin t-horizon and is fully realized.
        realized = np.log(close / data["open"].shift(horizon - 1).where(lambda x: x > 0))
        realized = realized / ((atr / close).shift(horizon) * np.sqrt(horizon)).replace(0, np.nan)
        weight_sum = pd.Series(0.0, index=data.index)
        first = weight_sum.copy()
        second = weight_sum.copy()
        # Exclude analogs whose outcome interval overlaps the current shape.
        for lag in range(horizon + length, horizon + length + params.history_bars):
            distance = ((area - area.shift(lag)) / params.bandwidth) ** 2
            distance += ((endpoint - endpoint.shift(lag)) / (2 * params.bandwidth)) ** 2
            outcome = realized.shift(lag - horizon).replace([np.inf, -np.inf], np.nan)
            weight = np.exp(-0.5 * distance).where(outcome.notna(), 0.0).fillna(0.0)
            outcome = outcome.fillna(0.0)
            weight_sum += weight
            first += weight * outcome
            second += weight * outcome * outcome
        denominator = weight_sum.replace(0, np.nan)
        mean = first / denominator
        variance = (second / denominator - mean * mean).clip(lower=0)
        edge = (mean / np.sqrt(variance.clip(lower=0.0625))).where(weight_sum >= 12, 0.0)
        return pd.DataFrame({
            "signature_area": area,
            "endpoint": endpoint,
            "atr": atr,
            "analog_weight": weight_sum,
            "forecast": mean,
            "edge": edge,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        edge = indicators["edge"].fillna(0.0).to_numpy(dtype=float)
        side = 0
        pending = False
        armed = False
        entry = 0.0
        entry_bar = 0
        stop = 0.0
        trade_size = 0.25
        warmup = self.warmup_bars(params)
        for i in range(n):
            if side:
                if pending:
                    # The previous close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    entry = opens[i]
                    entry_bar = i
                    stop = side * entry - params.initial_stop_atr * atr[i - 1]
                    pending = False
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                # Signed-price coordinates make the protective stop always rise,
                # including shorts, whose actual price stop moves downward.
                price = side * close[i]
                if price <= stop or i - entry_bar + 1 >= params.holding_bars:
                    side = 0
                    continue
                if side * (close[i] - entry) / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, price - params.trail_atr * atr[i])
                signals[i] = side
                sizes[i] = trade_size
            elif i >= warmup and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]) and close[i] > 0:
                if abs(edge[i]) >= params.edge_threshold:
                    side = 1 if edge[i] > 0 else -1
                    strength = np.clip((abs(edge[i]) - params.edge_threshold) / (2 * params.edge_threshold), 0.0, 1.0)
                    trade_size = float(0.25 + 0.70 * strength)
                    pending = True
                    armed = False
                    signals[i] = side
                    sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShapeParams(shape_bars=8, history_bars=60, atr_bars=5)
    strategy = GeneratedStrategy()
    start = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=start + 45, freq="B")
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    data = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1000000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-10], params)
    pd.testing.assert_frame_equal(full.iloc[:-10], prefix)
    for direction in (1, -1):
        test_data = data.copy()
        test_data["open"] = 100.0
        test_data["close"] = 100.0
        test_data.iloc[start:start + 7, test_data.columns.get_loc("close")] = (
            100 + direction * np.array([0, 0, 2, 4, 1, 1, 1])
        )
        test_ind = pd.DataFrame({"atr": 0.5, "edge": 0.0}, index=index)
        test_ind.iloc[start, 1] = direction * params.edge_threshold * 2
        result = strategy.generate_signals(test_data, test_ind, None, params).data
        assert result["signal"].iloc[start] == 0
        assert result["signal"].iloc[start + 1] == direction
        assert result["signal"].iloc[start + 4] == direction
        assert result["signal"].iloc[start + 5] == 0
        assert 0.25 < result["size"].iloc[start + 1] <= 0.95
        assert set(result["signal"].unique()) <= {-1, 0, 1}
        assert (result["size"] > 0).all()
    print("Causality, next-bar fills, long/short trailing exits, and sizing passed.")
