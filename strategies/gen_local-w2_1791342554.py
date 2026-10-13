from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 5
    history_bars: int = 504
    neighbors: int = 8
    hold_bars: int = 18
    atr_bars: int = 14
    forecast_threshold: float = 0.9
    minimum_vote: float = 0.75
    maximum_distance: float = 0.75
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791342554"

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return max(200, params.history_bars + params.hold_bars + params.shape_bars + params.atr_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if min(p.shape_bars, p.history_bars, p.neighbors, p.hold_bars, p.atr_bars) < 1:
            raise ValueError("Lookbacks, holding period and neighbor count must be positive.")
        if p.forecast_threshold <= 0 or p.maximum_distance <= 0 or p.trail_atr <= 0 or not 0.5 < p.minimum_vote <= 1:
            raise ValueError("Invalid forecast, similarity, voting or trailing parameters.")
        close = data["close"]
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars, min_periods=p.atr_bars).mean().replace(0, np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["gap"] = ((data["open"] - previous) / out["atr"].shift(1)).replace([np.inf, -np.inf], np.nan)
        columns = []
        for lag in range(p.shape_bars):
            name = f"gap_lag_{lag}"
            out[name] = out["gap"].shift(lag)
            columns.append(name)
        # This label is observable at its own row; candidate j uses row j + hold_bars.
        out["matured_outcome"] = ((close - close.shift(p.hold_bars)) / out["atr"].shift(p.hold_bars)).replace([np.inf, -np.inf], np.nan)
        out["forecast"] = np.nan
        out["vote"] = np.nan
        out["analog_distance"] = np.nan
        features = out[columns].to_numpy(dtype=float)
        outcomes = out["matured_outcome"].to_numpy(dtype=float)
        moving_average = out["ma200"].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        regimes = np.sign(prices - moving_average)
        valid = np.isfinite(features).all(axis=1) & np.isfinite(regimes)
        forecasts = np.full(len(data), np.nan)
        votes = np.full(len(data), np.nan)
        distances = np.full(len(data), np.nan)
        for t in range(200 + p.hold_bars, len(data)):
            if not valid[t] or regimes[t] == 0:
                continue
            end = t - p.hold_bars
            candidates = np.arange(max(0, end - p.history_bars + 1), end + 1)
            keep = valid[candidates] & (regimes[candidates] == regimes[t]) & np.isfinite(outcomes[candidates + p.hold_bars])
            candidates = candidates[keep]
            if len(candidates) < p.neighbors:
                continue
            squared = np.mean((features[candidates] - features[t]) ** 2, axis=1)
            selected = []
            # Disjoint outcome windows keep a single episode from supplying all votes.
            for local in np.argsort(squared):
                endpoint = candidates[local]
                if all(abs(endpoint - candidates[other]) >= p.hold_bars for other in selected):
                    selected.append(int(local))
                    if len(selected) == p.neighbors:
                        break
            if len(selected) < p.neighbors:
                continue
            labels = outcomes[candidates[selected] + p.hold_bars]
            forecasts[t] = float(np.median(labels))
            votes[t] = float(np.mean(np.sign(labels) == np.sign(forecasts[t])))
            distances[t] = float(np.sqrt(np.mean(squared[selected])))
        out["forecast"] = forecasts
        out["vote"] = votes
        out["analog_distance"] = distances
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        vote = indicators["vote"].to_numpy(dtype=float)
        distance = indicators["analog_distance"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        water = np.nan
        stop = np.nan
        for t in range(len(data)):
            price = prices[t]
            if not np.isfinite(price) or not np.isfinite(atr[t]) or atr[t] <= 0:
                position = 0
                age = 0
                water = stop = np.nan
                continue
            if position:
                # The first update is the first held bar, after the entry decision.
                age += 1
                if position == 1:
                    water = price if not np.isfinite(water) else max(water, price)
                    candidate_stop = water - p.trail_atr * atr[t]
                    stop = candidate_stop if not np.isfinite(stop) else max(stop, candidate_stop)
                    stopped = price <= stop
                else:
                    # Mirror the high-water trailing exit for short positions.
                    water = price if not np.isfinite(water) else min(water, price)
                    candidate_stop = water + p.trail_atr * atr[t]
                    stop = candidate_stop if not np.isfinite(stop) else min(stop, candidate_stop)
                    stopped = price >= stop
                if stopped or age >= p.hold_bars:
                    position = 0
                    age = 0
                    water = stop = np.nan
                signals[t] = position
                continue
            ready = np.isfinite(ma[t]) and np.isfinite(forecast[t]) and np.isfinite(vote[t]) and np.isfinite(distance[t])
            if ready and vote[t] >= p.minimum_vote and distance[t] <= p.maximum_distance:
                if price > ma[t] and forecast[t] >= p.forecast_threshold:
                    position = 1
                elif price < ma[t] and forecast[t] <= -p.forecast_threshold:
                    position = -1
            signals[t] = position
        df = pd.DataFrame({"signal": signals, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = AnalogParams()
    index = pd.date_range("2020-01-01", periods=32, freq="B")
    for direction in (1, -1):
        close = np.full(32, 100.0)
        close[:5] = [100, 100 + direction, 100 + 5 * direction, 100 + 4 * direction, 100 + direction]
        data = pd.DataFrame({"close": close}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "ma200": 100.0 - direction, "forecast": 0.0, "vote": 1.0, "analog_distance": 0.0}, index=index)
        indicators.iloc[0, indicators.columns.get_loc("forecast")] = 2.0 * direction
        result = GeneratedStrategy.generate_signals(data, indicators, None, p).data
        assert result["signal"].iloc[:6].tolist() == [0, direction, direction, direction, direction, 0]
        data["close"] = 100.0
        result = GeneratedStrategy.generate_signals(data, indicators, None, p).data
        assert (result["signal"].iloc[1:19] == direction).all()
        assert result["signal"].iloc[19] == 0
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
