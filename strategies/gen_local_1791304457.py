from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SilhouetteParams:
    atr_window: int = 14
    shape_bars: int = 5
    history_bars: int = 252
    neighbors: int = 12
    max_distance: float = 0.45
    min_edge: float = 0.003
    noise_ratio: float = 0.25
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[SilhouetteParams]):
    strategy_id = "gen_local_1791304457"

    @classmethod
    def params_type(cls):
        return SilhouetteParams

    def warmup_bars(self, params: SilhouetteParams) -> int:
        return int(params.atr_window + params.shape_bars + params.history_bars + params.holding_bars + 2)

    def indicators(self, data: pd.DataFrame, params: SilhouetteParams) -> pd.DataFrame:
        if min(params.atr_window, params.shape_bars, params.neighbors, params.holding_bars) < 1:
            raise ValueError("Lookbacks, neighbors, and holding_bars must be positive")
        if params.history_bars <= max(params.shape_bars, params.holding_bars + 1) + params.neighbors:
            raise ValueError("history_bars is too short for mature analogs")
        if not all(np.isfinite(x) for x in (params.max_distance, params.min_edge, params.noise_ratio, params.profit_target)):
            raise ValueError("Thresholds must be finite")
        if params.max_distance <= 0 or params.min_edge <= 0 or params.noise_ratio < 0 or params.profit_target <= 0:
            raise ValueError("Invalid thresholds")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        scale = out["atr"].where(out["atr"] > 0)
        body_top = data[["open", "close"]].max(axis=1)
        body_bottom = data[["open", "close"]].min(axis=1)
        out["upper"] = (data["high"] - body_top).clip(lower=0) / scale
        out["lower"] = (body_bottom - data["low"]).clip(lower=0) / scale
        out["body"] = (data["close"] - data["open"]) / scale
        features = []
        for lag in range(params.shape_bars):
            for component in ("upper", "lower", "body"):
                name = f"{component}_lag_{lag}"
                out[name] = out[component].shift(lag)
                features.append(name)
        matrix = out[features].to_numpy(dtype=float)
        valid = np.isfinite(matrix).all(axis=1)
        opens = data["open"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        dispersion = np.full(len(data), np.nan)
        analog_count = np.zeros(len(data), dtype=int)
        separation = max(params.shape_bars, params.holding_bars + 1)
        for i in range(len(data)):
            if not valid[i]:
                continue
            candidates = np.arange(max(0, i - params.history_bars), max(0, i - separation + 1))
            if len(candidates) == 0:
                continue
            entry_prices = opens[candidates + 1]
            exit_prices = opens[candidates + params.holding_bars + 1]
            eligible = valid[candidates] & np.isfinite(entry_prices) & np.isfinite(exit_prices) & (entry_prices > 0) & (exit_prices > 0)
            candidates = candidates[eligible]
            if len(candidates) < params.neighbors:
                continue
            distances = np.sqrt(np.mean((matrix[candidates] - matrix[i]) ** 2, axis=1))
            nearest = np.argsort(distances, kind="stable")[:params.neighbors]
            if distances[nearest[-1]] > params.max_distance:
                continue
            matched = candidates[nearest]
            outcomes = opens[matched + params.holding_bars + 1] / opens[matched + 1] - 1.0
            forecast[i] = float(np.mean(outcomes))
            dispersion[i] = float(np.std(outcomes))
            analog_count[i] = len(matched)
        out["forecast"] = forecast
        out["dispersion"] = dispersion
        out["analog_count"] = analog_count
        qualifies = out["forecast"].abs().ge(params.min_edge) & out["forecast"].abs().ge(params.noise_ratio * out["dispersion"])
        out["candidate"] = np.where(qualifies, np.sign(out["forecast"].fillna(0)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SilhouetteParams) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        confirmed = np.zeros(len(data), dtype=int)
        if len(data) > 1:
            confirmed[1:] = np.where((candidate[1:] != 0) & (candidate[1:] == candidate[:-1]), candidate[1:], 0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    target_hit = np.isfinite(gain) and gain >= params.profit_target
                    time_hit = i - entry_bar + 1 >= params.holding_bars
                    if target_hit or time_hit:
                        position = 0
            elif confirmed[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"candidate": [1, 1, 0, 0, 0, 0]}, index=index)
    params = SilhouetteParams(holding_bars=2)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    indicators["candidate"] = [-1, -1, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 97.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, 0, 0, 0]
