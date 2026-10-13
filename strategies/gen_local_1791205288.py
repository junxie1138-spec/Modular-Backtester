from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    trend_window: int = 12
    noise_window: int = 20
    breakout_window: int = 6
    atr_window: int = 14
    innovation_threshold: float = 0.35
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.trend_window, self.noise_window, self.breakout_window, self.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if min(self.innovation_threshold, self.breakeven_pct, self.trail_atr) <= 0:
            raise ValueError("Thresholds and stop distances must be positive.")
        if not 3 <= self.holding_bars <= 5:
            raise ValueError("Holding horizon must be three to five bars.")


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = "gen_local_1791205288"

    @classmethod
    def params_type(cls):
        return TrendInnovationParams

    @staticmethod
    def warmup_bars(params: TrendInnovationParams) -> int:
        return int(max(params.trend_window + 2, params.noise_window + 2,
                       params.breakout_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        n = params.trend_window
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = out["log_close"].diff()
        out["noise"] = out["log_return"].rolling(params.noise_window).std(ddof=0)
        out["trend_mean"] = out["log_close"].rolling(n).mean()
        t = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["slope"] = ((t * out["log_close"]).rolling(n).mean()
                        - t.rolling(n).mean() * out["trend_mean"]) / ((n * n - 1) / 12.0)
        out["trend_endpoint"] = out["trend_mean"] + out["slope"] * (n - 1) / 2.0
        out["forecast"] = out["trend_endpoint"].shift(1) + out["slope"].shift(1)
        scale = out["noise"].shift(1).where(out["noise"].shift(1) > 1e-12)
        out["forecast_innovation"] = (out["log_close"] - out["forecast"]) / scale
        out["slope_innovation"] = n * out["slope"].diff() / scale
        out["upper"] = data["high"].rolling(params.breakout_window).max().shift(1)
        out["lower"] = data["low"].rolling(params.breakout_window).min().shift(1)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TrendInnovationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        values = indicators[["atr", "slope", "slope_innovation",
                             "forecast_innovation", "upper", "lower"]].to_numpy(dtype=float)
        side, age = 0, 0
        entry, stop, extreme, locked_size = 0.0, 0.0, 0.0, 1.0
        armed = False
        for i in range(count):
            opening, high, low, closing = prices[i]
            atr, slope, score, surprise, upper, lower = values[i]
            valid_price = np.isfinite(prices[i]).all() and min(prices[i]) > 0
            if side:
                if not valid_price or not np.isfinite(atr) or atr <= 0:
                    side, age, armed = 0, 0, False
                else:
                    if age == 0:
                        # Yesterday's decision fills at today's open.
                        entry = opening
                        extreme = entry
                        stop = entry - side * params.trail_atr * values[i - 1, 0]
                        armed = False
                    age += 1
                    touched = low <= stop if side == 1 else high >= stop
                    if not touched:
                        extreme = max(extreme, high) if side == 1 else min(extreme, low)
                        armed = armed or side * (extreme / entry - 1.0) >= params.breakeven_pct
                        if armed:
                            candidate = extreme - side * params.trail_atr * atr
                            # Long stops rise; short buy-to-cover stops fall.
                            stop = max(stop, entry, candidate) if side == 1 else min(stop, entry, candidate)
                            touched = closing <= stop if side == 1 else closing >= stop
                    # Stop touches are exit decisions; execution is next open.
                    if touched or age >= params.holding_bars:
                        side, age, armed = 0, 0, False
            elif valid_price and np.isfinite(values[i]).all() and atr > 0:
                if closing > upper and slope > 0 and surprise > 0 and score >= params.innovation_threshold:
                    side = 1
                elif closing < lower and slope < 0 and surprise < 0 and score <= -params.innovation_threshold:
                    side = -1
                if side:
                    locked_size = float(np.clip(0.35 * abs(score) / params.innovation_threshold, 0.25, 0.95))
                    age = 0
            signals[i] = side
            sizes[i] = locked_size if side else 1.0

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = TrendInnovationParams()
    index = pd.date_range("2020-01-01", periods=6)
    base = pd.DataFrame([[100, 103, 99, 102], [102, 104, 102, 103],
                         [103, 103, 101, 101.5], [101.5, 102, 101, 101.5],
                         [101.5, 102, 101, 101.5], [101.5, 102, 101, 101.5]],
                        columns=["open", "high", "low", "close"], index=index, dtype=float)
    for direction in (1, -1):
        data = base.copy()
        if direction == -1:
            data["open"] = 200 - base["open"]
            data["close"] = 200 - base["close"]
            data["high"] = 200 - base["low"]
            data["low"] = 200 - base["high"]
        data["volume"] = 1000.0
        features = pd.DataFrame({"atr": 1.0, "slope": float(direction),
                                 "slope_innovation": [direction, 0, 0, 0, 0, 0],
                                 "forecast_innovation": float(direction),
                                 "upper": 101.0, "lower": 99.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]
        assert (result["size"] > 0).all()
        assert result["size"].iloc[1] == result["size"].iloc[2] == 0.95
        # An unarmed, untouched position expires after exactly four sessions.
        data.loc[index[1]:, ["open", "high", "low", "close"]] = [102, 102.5, 101.5, 102] if direction == 1 else [98, 98.5, 97.5, 98]
        result = GeneratedStrategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
