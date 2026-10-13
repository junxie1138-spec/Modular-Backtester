from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    block: int = 4
    history: int = 504
    neighbors: int = 12
    horizon: int = 18
    atr_period: int = 14
    edge: float = 0.008
    max_distance: float = 0.45
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1790908891"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(200, 20 + 2 * params.block, params.atr_period + 1, params.horizon + 1) + params.horizon + 3 * params.block + params.neighbors)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if min(params.block, params.history, params.neighbors, params.horizon, params.atr_period) < 1:
            raise ValueError("Lookbacks and neighbor count must be positive")
        if params.neighbors > params.history:
            raise ValueError("Neighbor count exceeds history")
        if not all(np.isfinite(x) and x > 0 for x in (params.edge, params.max_distance, params.trail_atr)):
            raise ValueError("Thresholds must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0) / close
        out["range_baseline"] = out["daily_range"].rolling(20, min_periods=20).mean()
        out["range_block"] = out["daily_range"].rolling(params.block, min_periods=params.block).mean()
        denominator = out["range_baseline"].where(out["range_baseline"] > 0)
        for lag in range(3):
            out[f"shape_{lag}"] = out["range_block"].shift(lag * params.block) / denominator
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["regime"] = np.where(out["ma200"].notna(), np.sign(close - out["ma200"]), 0)
        out["completed_return"] = close / close.shift(params.horizon) - 1.0
        out["forecast"] = np.nan
        out["match_distance"] = np.nan
        shapes = out[["shape_0", "shape_1", "shape_2"]].to_numpy(dtype=float)
        regimes = out["regime"].to_numpy(dtype=float)
        completed = out["completed_return"].to_numpy(dtype=float)
        forecasts = np.full(len(data), np.nan)
        distances = np.full(len(data), np.nan)
        valid_shape = np.isfinite(shapes).all(axis=1)
        separation = params.horizon + 3 * params.block
        # ponytail: bounded exhaustive search; use an index if history grows substantially.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not valid_shape[i] or not np.isfinite(regimes[i]) or regimes[i] == 0:
                continue
            end = i - separation + 1
            candidates = np.arange(max(0, end - params.history), end)
            outcomes = completed[candidates + params.horizon]
            eligible = valid_shape[candidates] & (regimes[candidates] == regimes[i]) & np.isfinite(outcomes)
            candidates = candidates[eligible]
            outcomes = outcomes[eligible]
            if len(candidates) < params.neighbors:
                continue
            distance = np.sqrt(np.mean((shapes[candidates] - shapes[i]) ** 2, axis=1))
            nearest = np.argsort(distance, kind="stable")[:params.neighbors]
            distances[i] = float(distance[nearest].mean())
            if distance[nearest].max() <= params.max_distance:
                forecasts[i] = float(np.median(outcomes[nearest]))
        out["forecast"] = forecasts
        out["match_distance"] = distances
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entered = -1
        watermark = np.nan
        stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(ma[i]):
                position = 0
                continue
            if position:
                if position == 1:
                    watermark = max(watermark, close[i])
                    stop = max(stop, watermark - params.trail_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    # Short mirror: the lowest close and stop only ratchet downward.
                    watermark = min(watermark, close[i])
                    stop = min(stop, watermark + params.trail_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entered >= params.horizon:
                    position = 0
                    continue
            elif np.isfinite(forecast[i]):
                if forecast[i] >= params.edge and close[i] > ma[i]:
                    position = 1
                elif forecast[i] <= -params.edge and close[i] < ma[i]:
                    position = -1
                if position:
                    entered = i
                    watermark = close[i]
                    stop = close[i] - position * params.trail_atr * atr[i]
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ShapeParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        prices = np.array([100., 105., 110., 109., 106., 100., 100., 100.])
        if direction == -1:
            prices = 200.0 - prices
        data = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.}, index=index)
        indicators = pd.DataFrame({"atr": 1., "ma200": 90. if direction == 1 else 110., "forecast": direction * 0.02}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, direction, direction]
        assert (result["size"] > 0).all()
