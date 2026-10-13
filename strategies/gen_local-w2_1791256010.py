from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    shape_bars: int = 6
    history_bars: int = 252
    neighbors: int = 12
    atr_bars: int = 14
    hold_bars: int = 8
    edge_atr: float = 0.35
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791256010"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(params.atr_bars + params.shape_bars + params.history_bars + params.hold_bars + 2)

    def indicators(self, data, params):
        p = params
        if min(p.shape_bars, p.history_bars, p.neighbors, p.atr_bars, p.hold_bars) < 1:
            raise ValueError("Lookbacks and neighbor count must be positive.")
        if p.shape_bars < 2 or p.history_bars <= p.hold_bars + p.shape_bars + p.neighbors:
            raise ValueError("Insufficient analog history.")
        if not np.isfinite(p.edge_atr) or not np.isfinite(p.stop_atr) or min(p.edge_atr, p.stop_atr) <= 0:
            raise ValueError("Thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        scale = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = (data["open"] - previous) / scale
        out["body"] = (data["close"] - data["open"]) / scale
        g, b = out["gap"], out["body"]
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of paired daily gap/body increments.
        for lag in range(1, p.shape_bars):
            cross = g.shift(lag) * b - b.shift(lag) * g
            area = area + cross.rolling(p.shape_bars - lag, min_periods=p.shape_bars - lag).sum()
        out["area"] = 0.5 * area / p.shape_bars
        out["gap_net"] = g.rolling(p.shape_bars, min_periods=p.shape_bars).sum() / np.sqrt(p.shape_bars)
        out["body_net"] = b.rolling(p.shape_bars, min_periods=p.shape_bars).sum() / np.sqrt(p.shape_bars)
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[["area", "gap_net", "body_net"]].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = out["atr"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        # ponytail: bounded exact neighbor scan; index features if histories become much larger.
        for i in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # Each outcome ends before the current shape begins.
            last = i - p.hold_bars - p.shape_bars
            candidates = np.arange(max(0, i - p.history_bars), last + 1)
            valid = (
                np.isfinite(features[candidates]).all(axis=1)
                & np.isfinite(atr[candidates]) & (atr[candidates] > 0)
                & np.isfinite(opens[candidates + 1])
                & np.isfinite(closes[candidates + p.hold_bars])
                & (features[candidates, 0] * features[i, 0] > 0)
            )
            candidates = candidates[valid]
            if len(candidates) < p.neighbors:
                continue
            delta = features[candidates] - features[i]
            distance = 4.0 * delta[:, 0] ** 2 + 0.25 * (delta[:, 1:] ** 2).sum(axis=1)
            chosen = candidates[np.argsort(distance, kind="stable")[:p.neighbors]]
            # No future-index access beyond i; label starts at the next open.
            outcomes = (closes[chosen + p.hold_bars] - opens[chosen + 1]) / atr[chosen]
            forecast[i] = float(np.median(outcomes))
        out["forecast"] = forecast
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        forecast = indicators["forecast"].to_numpy(dtype=float)
        proposal = np.where(np.isfinite(forecast) & (forecast > p.edge_atr), 1,
                            np.where(np.isfinite(forecast) & (forecast < -p.edge_atr), -1, 0))
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        entry_price, frozen_distance = np.nan, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # This open executes yesterday's already-confirmed decision.
                    entry_price = opens[i]
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                stopped = not invalid and side * (closes[i] - entry_price) < -frozen_distance
                expired = i - entry_bar + 1 >= p.hold_bars
                if invalid or stopped or expired:
                    side = 0
            elif (
                i > 0 and proposal[i] != 0 and proposal[i] == proposal[i - 1]
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                side = int(proposal[i])
                entry_bar = i + 1
                entry_price = np.nan
                frozen_distance = p.stop_atr * atr[i]
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        sample = data.copy()
        sample.iloc[2:5, sample.columns.get_loc("close")] = 100.0 + side * np.array([5.0, -1.0, -3.0])
        ind = pd.DataFrame({"atr": 1.0, "forecast": [float(side), float(side)] + [0.0] * 12}, index=index)
        result = strategy.generate_signals(sample, ind, ctx, SignatureParams()).data
        assert result["signal"].tolist() == [0, 0, side, side, side] + [0] * 9
        assert (result["size"] > 0).all()
        result = strategy.generate_signals(data, ind, ctx, SignatureParams(hold_bars=5)).data
        assert result["signal"].tolist() == [0, 0] + [side] * 5 + [0] * 7
    n = 420
    t = np.arange(n, dtype=float)
    close = 100.0 + 0.04 * t + 2.0 * np.sin(t / 7.0)
    opening = close + 0.3 * np.cos(t / 5.0)
    sample = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 1.0,
                           "low": np.minimum(opening, close) - 1.0,
                           "close": close, "volume": 1000.0},
                          index=pd.date_range("2020-01-01", periods=n))
    params = SignatureParams()
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:360], params)
    assert full.iloc[:360].equals(prefix)
    assert full["forecast"].notna().any()


if __name__ == "__main__":
    _self_check()
