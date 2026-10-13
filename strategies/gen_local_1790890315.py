from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    analog_window: int = 252
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790890315"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.analog_window) + 15

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.analog_window < 8 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("analog_window must be >= 8 and trail_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["volume_step"] = np.log(data["volume"].where(data["volume"] > 0)).diff()
        r, v = out["return"], out["volume_step"]
        out["price_scale"] = r.abs().rolling(6, min_periods=6).sum()
        out["volume_scale"] = v.abs().rolling(6, min_periods=6).sum()
        area = pd.Series(0.0, index=data.index)
        # Half the antisymmetric second-order signature of the price-volume path.
        for lag in range(1, 6):
            area += (r.shift(lag) * v - v.shift(lag) * r).rolling(6 - lag, min_periods=6 - lag).sum()
        out["area"] = 0.5 * area / (out["price_scale"] * out["volume_scale"]).replace(0.0, np.nan)
        out["price_endpoint"] = r.rolling(6, min_periods=6).sum() / out["price_scale"].replace(0.0, np.nan)
        out["volume_endpoint"] = v.rolling(6, min_periods=6).sum() / out["volume_scale"].replace(0.0, np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        opening = data["open"].where(data["open"] > 0)
        out["realized_open_return"] = opening / opening.shift(2) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["area", "price_endpoint", "volume_endpoint"]].to_numpy()
        realized = out["realized_open_return"].to_numpy()
        forecast = np.full(len(data), np.nan)
        direction = np.zeros(len(data), dtype=int)
        # ponytail: O(bars * window) search; use indexed neighbors if histories become large.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            history = np.arange(i - params.analog_window - 8, i - 8)
            valid = np.isfinite(features[history]).all(axis=1) & np.isfinite(realized[history + 3])
            history = history[valid]
            if len(history) < 8:
                continue
            distance = ((features[history] - features[i]) ** 2).sum(axis=1)
            neighbors = history[np.argsort(distance, kind="stable")[:8]]
            # Shape ends at s; entry is open[s+1], two-session exit is open[s+3].
            outcomes = realized[neighbors + 3]
            forecast[i] = float(np.median(outcomes))
            side = int(np.sign(forecast[i]))
            if abs(forecast[i]) > 0.0015 and np.count_nonzero(outcomes * side > 0) >= 6:
                direction[i] = side
        out["forecast"] = forecast
        out["direction"] = direction
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, stop = np.nan, -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["direction"].to_numpy(dtype=float)
        for i, close in enumerate(closes):
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position, age = 0, 0
                high_water, stop = np.nan, -np.inf
            elif position:
                # First in-trade close occurs one bar after the entry decision.
                high_water = close if not np.isfinite(high_water) else max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                age += 1
                # Required downward high-water stop applies to both long and short positions.
                if close < stop or age >= 2:
                    position, age = 0, 0
                    high_water, stop = np.nan, -np.inf
            elif np.isfinite(directions[i]) and directions[i] in (-1, 1):
                position = int(directions[i])
                age = 0
                high_water, stop = np.nan, -np.inf
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams(analog_window=8)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100.0, 102.0, 99.0, 100.0, 101.0, 100.0]}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "direction": [1, 0, 0, -1, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
