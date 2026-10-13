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
    neighbors: int = 12
    forecast_edge: float = 0.008
    pressure_threshold: float = 0.12
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791293096"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.shape_bars + params.history_bars + params.hold_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        if (p.shape_bars < 2 or p.history_bars < p.neighbors
                or p.neighbors < 1 or p.hold_bars < 1
                or p.forecast_edge <= 0 or not 0 < p.pressure_threshold < 1
                or p.profit_target <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_return"] = np.log(close).diff()
        out["path_variation"] = out["log_return"].abs().rolling(p.shape_bars).sum()
        scale = out["path_variation"].replace(0.0, np.nan)
        weighted = pd.Series(0.0, index=data.index)
        for lag in range(p.shape_bars):
            weighted += (p.shape_bars - 1 - 2 * lag) * out["log_return"].shift(lag)
        # Antisymmetric second-order signature of normalized time and log price.
        out["signature_area"] = weighted / (2.0 * p.shape_bars * scale)
        out["displacement"] = out["log_return"].rolling(p.shape_bars).sum() / scale
        span = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = ((2 * close - data["high"] - data["low"]) / span).clip(-1, 1)
        out["pressure"] = out["close_location"].rolling(p.shape_bars).mean()
        features = out[["signature_area", "displacement"]].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        # ponytail: bounded linear scan per bar; index neighbors if history grows materially.
        for i in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Labels and source paths end before the current query path begins.
            last = i - p.hold_bars - p.shape_bars
            first = max(p.shape_bars, last - p.history_bars + 1)
            candidates = np.arange(first, last + 1)
            labels = prices[candidates + p.hold_bars] / opens[candidates + 1] - 1.0
            valid = (np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(labels) & (opens[candidates + 1] > 0))
            candidates, labels = candidates[valid], labels[valid]
            if len(candidates) < p.neighbors:
                continue
            delta = features[candidates] - features[i]
            distance = delta[:, 0] ** 2 + 0.25 * delta[:, 1] ** 2
            nearest = np.argsort(distance, kind="stable")[:p.neighbors]
            forecast[i] = float(np.median(labels[nearest]))
        out["analog_forecast"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        p = params
        forecast = indicators["analog_forecast"]
        pressure = indicators["pressure"]
        valid = np.isfinite(forecast) & np.isfinite(pressure)
        desired = np.select(
            [valid & (forecast >= p.forecast_edge) & (pressure >= p.pressure_threshold),
             valid & (forecast <= -p.forecast_edge) & (pressure <= -p.pressure_threshold)],
            [1, -1], default=0,
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, decision_bar = 0, -1
        entry = np.nan
        for i in range(len(data)):
            if side:
                if i == decision_bar + 1:
                    entry = opens[i]
                gain = side * (closes[i] / entry - 1.0) if np.isfinite(entry) and entry > 0 else np.nan
                expired = i - decision_bar >= p.hold_bars
                if expired or (np.isfinite(gain) and gain >= p.profit_target) or not np.isfinite(entry) or entry <= 0:
                    side = 0
                raw[i] = side
                # Exit occupies this decision bar; reentry requires a later decision.
                continue
            if i >= self.warmup_bars(p) - 1 and desired[i]:
                side, decision_bar, entry = int(desired[i]), i, np.nan
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    p = ShapeParams(shape_bars=2, history_bars=4, neighbors=1, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=30)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    decision = strategy.warmup_bars(p) - 1
    for side in (1, -1):
        ind = pd.DataFrame({"analog_forecast": 0.0, "pressure": 0.0}, index=index)
        ind.iloc[decision] = [side * 0.02, side * 0.5]
        timed = strategy.generate_signals(data, ind, ctx, p).data
        expected = np.zeros(len(data), dtype=int)
        expected[decision + 1:decision + 1 + p.hold_bars] = side
        assert np.array_equal(timed["signal"].to_numpy(), expected)
        target_data = data.copy()
        target_data.iloc[decision + 1, target_data.columns.get_loc("close")] = 100 * (1 + side * 0.05)
        target = strategy.generate_signals(target_data, ind, ctx, p).data
        assert target["signal"].iloc[decision + 1] == side
        assert target["signal"].iloc[decision + 2] == 0
        ind.iloc[decision, ind.columns.get_loc("pressure")] *= -1
        assert not strategy.generate_signals(data, ind, ctx, p).data["signal"].any()
        assert (timed["size"] > 0).all()
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=160)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    sample = pd.DataFrame({"open": close, "high": close * 1.01,
                           "low": close * 0.99, "close": close,
                           "volume": 1000.0}, index=index)
    full = strategy.indicators(sample, p)
    prefix = strategy.indicators(sample.iloc[:100], p)
    pd.testing.assert_frame_equal(full.iloc[:100], prefix)


if __name__ == "__main__":
    _self_check()
