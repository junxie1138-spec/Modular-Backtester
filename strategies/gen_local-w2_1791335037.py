from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    shape_window: int = 12
    history_bars: int = 252
    neighbors: int = 15
    holding_bars: int = 18
    min_edge: float = 0.012
    min_win_rate: float = 0.60
    max_distance: float = 0.15
    profit_target: float = 0.04
    size: float = 0.95

    def __post_init__(self):
        if not (self.shape_window >= 2 and self.history_bars >= self.neighbors >= 1 and self.holding_bars >= 1):
            raise ValueError("Invalid lookbacks or neighbor count")
        if not (self.min_edge >= 0 and 0 <= self.min_win_rate <= 1 and self.max_distance > 0 and self.profit_target > 0 and 0 < self.size <= 1):
            raise ValueError("Invalid thresholds or size")


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791335037"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(200, 2 * params.shape_window + params.holding_bars + params.history_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        w = p.shape_window
        close = data["close"].where(data["close"] > 0)
        log_return = np.log(close).diff()
        variation = log_return.abs().rolling(w, min_periods=w).sum()
        denominator = variation.where(variation > 0)
        # Second-order antisymmetric signature of (normalized time, log price).
        area = pd.Series(0.0, index=data.index)
        for lag in range(w):
            area = area + ((w - 1 - 2 * lag) / (2.0 * w)) * log_return.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["log_return"] = log_return
        out["path_variation"] = variation
        out["signature_area"] = area / denominator
        out["efficiency"] = log_return.rolling(w, min_periods=w).sum() / denominator
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        # At maturity m, this labels a historical entry at open[m-H+1].
        entry_open = data["open"].shift(p.holding_bars - 1)
        out["mature_return"] = close / entry_open.where(entry_open > 0) - 1.0
        a = out["signature_area"].to_numpy()
        e = out["efficiency"].to_numpy()
        mature = out["mature_return"].to_numpy()
        regime = (close > out["ma200"]).to_numpy()
        forecasts = np.full(len(data), np.nan)
        wins = np.full(len(data), np.nan)
        distances = np.full(len(data), np.nan)
        # ponytail: bounded O(bars * history_bars) scan; batch distances if histories become large.
        for i in range(self.warmup_bars(p), len(data)):
            if not (np.isfinite(a[i]) and np.isfinite(e[i])):
                continue
            end = i - w - p.holding_bars
            candidates = np.arange(end - p.history_bars + 1, end + 1)
            outcomes = mature[candidates + p.holding_bars]
            valid = np.isfinite(a[candidates]) & np.isfinite(e[candidates]) & np.isfinite(outcomes) & regime[candidates]
            candidates = candidates[valid]
            outcomes = outcomes[valid]
            if len(candidates) < p.neighbors:
                continue
            distance = (a[candidates] - a[i]) ** 2 + 0.25 * (e[candidates] - e[i]) ** 2
            nearest = np.argpartition(distance, p.neighbors - 1)[:p.neighbors]
            forecasts[i] = outcomes[nearest].mean()
            wins[i] = (outcomes[nearest] > 0).mean()
            distances[i] = np.sqrt(distance[nearest].mean())
        out["analog_return"] = forecasts
        out["analog_win_rate"] = wins
        out["analog_distance"] = distances
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        p = params
        eligible = (
            (data["close"] > indicators["ma200"])
            & (indicators["analog_return"] >= p.min_edge)
            & (indicators["analog_win_rate"] >= p.min_win_rate)
            & (indicators["analog_distance"] <= p.max_distance)
        ).fillna(False).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        holding = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(self.warmup_bars(p), len(data)):
            if holding:
                # A close decision on i-1 first becomes a position at open[i].
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + p.profit_target)
                expired = i - entry_bar + 1 >= p.holding_bars
                if target_hit or expired or not (np.isfinite(entry_price) and entry_price > 0):
                    holding = False
                    entry_bar = -1
                    entry_price = np.nan
            elif eligible[i]:
                holding = True
            raw[i] = int(holding)
        df = pd.DataFrame({"signal": raw, "size": float(p.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = ShapeParams()
    strategy = GeneratedStrategy()
    w = strategy.warmup_bars(p)
    n = w + 80
    index = pd.date_range("2000-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.004 * np.sin(np.arange(n) / 7.0)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:-20], p)
    pd.testing.assert_frame_equal(full.iloc[:-20], prefix)
    pd.testing.assert_frame_equal(strategy.generate_signals(data, full, None, p).data.iloc[:-20], strategy.generate_signals(data.iloc[:-20], prefix, None, p).data)
    data.loc[:, ["open", "close"]] = 100.0
    forced = pd.DataFrame({"ma200": 90.0, "analog_return": np.nan, "analog_win_rate": 1.0, "analog_distance": 0.0}, index=index)
    forced.iloc[w, forced.columns.get_loc("analog_return")] = 0.02
    timed = strategy.generate_signals(data, forced, None, p).data["signal"]
    assert timed.iloc[w] == 0 and timed.iloc[w + 1:w + p.holding_bars + 1].eq(1).all()
    assert timed.iloc[w + p.holding_bars + 1] == 0
    data.iloc[w + 4, data.columns.get_loc("close")] = 105.0
    targeted = strategy.generate_signals(data, forced, None, p).data["signal"]
    assert targeted.iloc[w + 4] == 1 and targeted.iloc[w + 5] == 0
    assert set(targeted.unique()) <= {0, 1}


if __name__ == "__main__":
    _self_check()
