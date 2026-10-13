from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 4
    history_bars: int = 252
    neighbors: int = 6
    horizon: int = 7
    atr_bars: int = 14
    min_edge: float = 0.003
    min_win: float = 0.60
    max_distance: float = 0.80
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791329751"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars + params.horizon + params.shape_bars + params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if min(p.shape_bars, p.history_bars, p.neighbors, p.horizon, p.atr_bars, p.max_hold) < 1:
            raise ValueError("Lookbacks, neighbor count and holding limit must be positive.")
        if p.min_edge < 0 or not 0 <= p.min_win <= 1 or p.max_distance <= 0 or p.breakeven_pct <= 0 or p.trail_atr <= 0:
            raise ValueError("Invalid forecast or stop parameters.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["true_range"] = tr
        out["atr"] = tr.rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        scale = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = ((data["open"] - previous) / scale).replace([np.inf, -np.inf], np.nan)
        shape_columns = []
        for width in range(1, p.shape_bars + 1):
            name = f"gap_path_{width}"
            out[name] = out["gap"].rolling(width, min_periods=width).sum()
            shape_columns.append(name)
        path = out[shape_columns]
        distances, outcomes = [], []
        # Candidate outcome intervals do not overlap; every outcome ends before today's shape starts.
        for lag in range(p.horizon + p.shape_bars, p.history_bars + p.horizon + p.shape_bars, p.horizon):
            dname, rname = f"distance_{lag}", f"outcome_{lag}"
            out[dname] = (path - path.shift(lag)).pow(2).mean(axis=1, skipna=False)
            historical_open = data["open"].shift(lag - 1)
            out[rname] = data["close"].shift(lag - p.horizon) / historical_open.where(historical_open > 0) - 1.0
            distances.append(out[dname].to_numpy())
            outcomes.append(out[rname].to_numpy())
        distance = np.column_stack(distances)
        outcome = np.column_stack(outcomes)
        valid = np.isfinite(distance) & np.isfinite(outcome)
        distance = np.where(valid, distance, np.inf)
        count = min(p.neighbors, distance.shape[1])
        selection = np.argpartition(distance, count - 1, axis=1)[:, :count]
        nearest = np.take_along_axis(distance, selection, axis=1)
        returns = np.take_along_axis(outcome, selection, axis=1)
        usable = np.isfinite(nearest) & np.isfinite(returns)
        weights = np.where(usable, 1.0 / (1.0 + nearest), 0.0)
        returns = np.where(usable, returns, 0.0)
        total = weights.sum(axis=1)
        denominator = np.where(total > 0, total, 1.0)
        complete = usable.sum(axis=1) == count
        out["forecast"] = np.where(complete, (weights * returns).sum(axis=1) / denominator, np.nan)
        out["hit_rate"] = np.where(complete, (weights * (returns > 0)).sum(axis=1) / denominator, np.nan)
        out["rms"] = np.where(complete, np.sqrt((weights * returns ** 2).sum(axis=1) / denominator), np.nan)
        safe_distance = np.where(usable, nearest, 0.0)
        out["match_distance"] = np.where(complete, (weights * safe_distance).sum(axis=1) / denominator, np.nan)
        strength = ((out["forecast"] - p.min_edge) / out["rms"].clip(lower=1e-8)).clip(0.0, 1.0)
        out["conviction"] = (strength * out["hit_rate"] * np.exp(-out["match_distance"])).fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = (
            (indicators["forecast"] >= p.min_edge)
            & (indicators["hit_rate"] >= p.min_win)
            & (indicators["match_distance"] <= p.max_distance)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy()
        conviction = indicators["conviction"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        pending = holding = armed = False
        entry = peak = stop = 0.0
        age = 0
        trade_size = 0.25
        for i in range(n):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    holding = True
                    entry = peak = opens[i]
                    stop = entry - p.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
            if holding:
                age += 1
                size[i] = trade_size
                breached = (np.isfinite(lows[i]) and lows[i] <= stop) or (np.isfinite(closes[i]) and closes[i] <= stop)
                if not breached:
                    if np.isfinite(highs[i]):
                        peak = max(peak, highs[i])
                    armed = armed or peak >= entry * (1.0 + p.breakeven_pct)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - p.trail_atr * atr[i])
                        # A newly raised stop can exit at the close, never retroactively at today's low.
                        breached = np.isfinite(closes[i]) and closes[i] <= stop
                if breached or age >= p.max_hold:
                    holding = False
                else:
                    signal[i] = 1
                continue
            if i >= self.warmup_bars(p) and ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                trade_size = float(0.25 + 0.70 * conviction[i])
                size[i] = trade_size
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, p = GeneratedStrategy(), AnalogParams()
    x = np.arange(650, dtype=float)
    close = 100.0 * np.exp(0.0003 * x + 0.012 * np.sin(x / 6.0))
    opening = close * (1.0 + 0.003 * np.sin(x / 3.0))
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.8,
        "low": np.minimum(opening, close) - 0.8, "close": close,
        "volume": np.full(len(x), 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=len(x)))
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(p))
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:600], p)
    assert np.allclose(full.iloc[:600], prefix, equal_nan=True)
    signals = strategy.generate_signals(data, full, ctx, p).data
    earlier = strategy.generate_signals(data.iloc[:600], prefix, ctx, p).data
    assert signals.iloc[:600].equals(earlier)
    assert signals["signal"].isin([0, 1]).all() and signals["size"].between(0.25, 0.95).all()
    assert signals["signal"].iloc[:strategy.warmup_bars(p) + 1].eq(0).all()
    p = AnalogParams(shape_bars=2, history_bars=14, horizon=2, atr_bars=3)
    w = strategy.warmup_bars(p)
    data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=w + 12))
    mock = pd.DataFrame({"atr": 1.0, "forecast": 0.0, "hit_rate": 1.0, "match_distance": 0.0, "conviction": 1.0}, index=data.index)
    mock.iloc[w, mock.columns.get_loc("forecast")] = 0.02
    for offset, high, low, close in [(1, 102.0, 100.0, 101.8), (2, 104.0, 101.5, 103.0), (3, 104.0, 102.2, 103.0), (4, 103.0, 101.9, 102.5)]:
        data.iloc[w + offset, data.columns.get_indexer(["high", "low", "close"])] = [high, low, close]
    mock.iloc[w + 3:w + 5, mock.columns.get_loc("atr")] = 10.0
    result = strategy.generate_signals(data, mock, ctx, p).data
    assert result["signal"].iloc[w] == 0
    assert result["signal"].iloc[w + 1:w + 5].eq(1).all()
    assert result["signal"].iloc[w + 5] == 0
    assert result["size"].iloc[w + 1] == 0.95


if __name__ == "__main__":
    _self_check()
