from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 5
    shape_points: int = 6
    noise_bars: int = 30
    history_bars: int = 504
    neighbors: int = 7
    hold_bars: int = 18
    atr_bars: int = 14
    entry_edge: float = 0.35
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791229495"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        p = params
        if min(p.roc_bars, p.shape_points, p.noise_bars, p.history_bars,
               p.neighbors, p.hold_bars, p.atr_bars) < 1:
            raise ValueError("Lookbacks, neighbors and holding period must be positive.")
        if p.noise_bars < 2 or p.neighbors < 2:
            raise ValueError("Noise window and neighbors must be at least two.")
        if p.history_bars < p.neighbors * p.hold_bars:
            raise ValueError("History must accommodate nonoverlapping analog outcomes.")
        if not np.isfinite(p.entry_edge) or not np.isfinite(p.stop_atr):
            raise ValueError("Thresholds must be finite.")
        if p.entry_edge <= 0 or p.stop_atr <= 0:
            raise ValueError("Thresholds must be positive.")
        return int(max(p.atr_bars + 1, p.roc_bars + 1 + p.noise_bars)
                   + 4 * (p.shape_points - 1) + p.history_bars + p.hold_bars)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        warmup = self.warmup_bars(p)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["atr"] = tr.rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        out["roc"] = close.pct_change(p.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["noise"] = out["acceleration"].rolling(
            p.noise_bars, min_periods=p.noise_bars).std(ddof=0)
        out["normalized_acceleration"] = (
            out["acceleration"] / out["noise"].where(out["noise"] > 0))
        columns = []
        for lag in range(p.shape_points):
            name = f"shape_{lag}"
            out[name] = out["normalized_acceleration"].shift(2 * lag)
            columns.append(name)
        features = out[columns].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        atr = out["atr"].to_numpy(dtype=float)
        edge = np.full(len(data), np.nan)
        similarity = np.full(len(data), np.nan)
        forecast = np.full(len(data), np.nan)
        span = 2 * (p.shape_points - 1)
        # ponytail: bounded historical scan; batch distances if intraday scale is needed.
        for t in range(warmup - 1, len(data)):
            if not np.isfinite(features[t]).all():
                continue
            latest = t - p.hold_bars - span
            candidates = np.arange(max(0, latest - p.history_bars + 1),
                                   latest + 1, p.hold_bars)
            valid = (np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(prices[candidates])
                     & np.isfinite(prices[candidates + p.hold_bars])
                     & np.isfinite(atr[candidates]) & (atr[candidates] > 0))
            candidates = candidates[valid]
            if len(candidates) < p.neighbors:
                continue
            distances = np.mean((features[candidates] - features[t]) ** 2, axis=1)
            chosen = np.argsort(distances)[:p.neighbors]
            analogs = candidates[chosen]
            weights = 1.0 / (0.1 + distances[chosen])
            weights /= weights.sum()
            outcomes = ((prices[analogs + p.hold_bars] - prices[analogs])
                        / atr[analogs])
            mean = np.sum(weights * outcomes)
            dispersion = np.sqrt(np.sum(weights * (outcomes - mean) ** 2))
            forecast[t] = mean
            edge[t] = mean / (0.5 + dispersion)
            similarity[t] = 1.0 / (1.0 + np.sum(weights * distances[chosen]))
        out["forecast_atr"] = forecast
        out["edge"] = edge
        out["similarity"] = similarity
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        edge = indicators["edge"].to_numpy(dtype=float)
        similarity = indicators["similarity"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        side, entry_bar = 0, -1
        entry_price, entry_atr, position_size = np.nan, np.nan, 0.15
        for t in range(len(data)):
            exited = False
            if side and t == entry_bar:
                # The prior close's decision fills at this bar's open.
                entry_price = opens[t]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side, exited = 0, True
            if side and t >= entry_bar:
                stopped = (np.isfinite(closes[t])
                           and side * (closes[t] - entry_price) < -p.stop_atr * entry_atr)
                expired = t - entry_bar + 1 >= p.hold_bars
                if stopped or expired:
                    side, exited = 0, True
            if not side and not exited and t + 1 < len(data):
                if (np.isfinite(edge[t]) and np.isfinite(similarity[t])
                        and np.isfinite(atr[t]) and atr[t] > 0
                        and abs(edge[t]) >= p.entry_edge):
                    side = 1 if edge[t] > 0 else -1
                    entry_bar = t + 1
                    entry_atr = atr[t]
                    position_size = float(0.15 + 0.80 * min(abs(edge[t]), 1.0)
                                          * np.clip(similarity[t], 0.0, 1.0))
            signals[t] = side
            sizes[t] = position_size if side else 0.15
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": [100, 100, 97, 100, 100, 100, 103, 100, 100],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0,
                               "edge": [0.5, 0, 0, 0, -1, 0, 0, 0, 0],
                               "similarity": 1.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(
        data, indicators, None, AnalogParams(stop_atr=2.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.55)
    assert np.isclose(result["size"].iloc[5], 0.95)
    data["close"] = 100.0
    indicators["edge"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(
        data, indicators, None, AnalogParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
