from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    slope_window: int = 6
    vol_window: int = 20
    shape_bars: int = 5
    archive_bars: int = 252
    neighbors: int = 7
    forecast_bars: int = 8
    edge_min: float = 0.35
    analog_radius: float = 1.4
    atr_window: int = 14
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791325700"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        p = params
        return int(max(p.vol_window + 1, p.slope_window + 2, p.atr_window + 1)
                   + 2 * p.shape_bars + p.forecast_bars + p.archive_bars)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if min(p.slope_window, p.vol_window, p.shape_bars, p.archive_bars,
               p.neighbors, p.forecast_bars, p.atr_window, p.max_hold_bars) < 1:
            raise ValueError("Lookbacks, neighbors and holding period must be positive")
        if p.vol_window < 2 or p.neighbors > p.archive_bars:
            raise ValueError("Require vol_window >= 2 and neighbors <= archive_bars")
        if min(p.initial_stop_atr, p.breakeven_pct, p.trail_atr, p.analog_radius) <= 0:
            raise ValueError("Stop and distance parameters must be positive")
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff()
        slope = returns.rolling(p.slope_window, min_periods=p.slope_window).mean()
        volatility = returns.rolling(p.vol_window, min_periods=p.vol_window).std()
        volatility = volatility.where(volatility > 1e-12)
        innovation = (returns - slope.shift(1)) / volatility
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(p.atr_window, min_periods=p.atr_window).mean()
        shape = pd.concat([innovation.shift(j) for j in range(p.shape_bars)], axis=1)
        current_shape = shape.to_numpy(dtype=float)
        matured_outcome = ((close / close.shift(p.forecast_bars) - 1.0)
                           / (volatility.shift(p.forecast_bars)
                              * np.sqrt(p.forecast_bars)))
        n = len(data)
        distances = np.full((n, p.archive_bars), np.inf)
        outcomes = np.full((n, p.archive_bars), np.nan)
        # Analog outcomes finish before the current shape begins.
        gap = p.forecast_bars + p.shape_bars
        for column, offset in enumerate(range(gap, gap + p.archive_bars)):
            difference = current_shape - shape.shift(offset).to_numpy(dtype=float)
            distance = np.sum(difference * difference, axis=1) / p.shape_bars
            outcome = matured_outcome.shift(offset - p.forecast_bars).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[:, column] = np.where(valid, distance, np.inf)
            outcomes[:, column] = outcome
        nearest = np.argpartition(distances, p.neighbors - 1, axis=1)[:, :p.neighbors]
        selected_distances = np.take_along_axis(distances, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        score = np.where(complete, selected_outcomes.mean(axis=1), np.nan)
        hit_rate = np.where(complete, (selected_outcomes > 0).mean(axis=1), np.nan)
        radius = np.where(complete, np.sqrt(selected_distances.mean(axis=1)), np.nan)
        result = pd.DataFrame({
            "log_return": returns,
            "latent_slope": slope,
            "volatility": volatility,
            "innovation": innovation,
            "true_range": true_range,
            "atr": atr,
            "analog_score": score,
            "analog_hit_rate": hit_rate,
            "analog_radius": radius
        }, index=data.index)
        for j in range(p.shape_bars):
            result[f"shape_{j}"] = shape.iloc[:, j]
        result["setup"] = (
            (result["innovation"] > 0)
            & (result["analog_score"] >= p.edge_min)
            & (result["analog_hit_rate"] >= 0.6)
            & (result["analog_radius"] <= p.analog_radius)
            & (result["atr"] > 0)
        ).fillna(False)
        result.iloc[:self.warmup_bars(p), result.columns.get_loc("setup")] = False
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        confirmed = (indicators["setup"].fillna(False).astype(bool)
                     & indicators["setup"].shift(1, fill_value=False).fillna(False).astype(bool))
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = high_water = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    entry = opens[i]
                    stop = entry - p.initial_stop_atr * atrs[i - 1]
                    high_water = entry
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                # A breached stop requests a next-open exit, as required by the contract.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                if np.isfinite(highs[i]):
                    high_water = max(high_water, highs[i])
                    armed = armed or high_water >= entry * (1.0 + p.breakeven_pct)
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, high_water - p.trail_atr * atrs[i])
                breached = breached or (np.isfinite(closes[i]) and closes[i] <= stop)
                if breached or age >= p.max_hold_bars:
                    active = False
                    continue
                raw[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [100.5, 100.5, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0],
        "low": [99.5, 99.5, 99.5, 100.5, 100.5, 100.5, 100.5, 100.5],
        "close": [100.0, 100.0, 100.5, 101.5, 102.5, 101.0, 101.0, 101.0],
        "volume": [1000000.0] * 8
    }, index=index)
    features = pd.DataFrame({
        "setup": [True, True, False, False, False, False, False, False],
        "atr": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    }, index=index)
    signals = strategy.generate_signals(data, features, None, params).data
    # The raised stop stays at 100 despite ATR expansion on bar four.
    assert signals["signal"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    features.loc[index[4]:, "atr"] = 1.0
    signals = strategy.generate_signals(data, features, None, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    features["setup"] = [True, False, False, False, False, False, False, False]
    assert not strategy.generate_signals(data, features, None, params).data["signal"].any()
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 400)))
    sample = pd.DataFrame({
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": np.full(400, 1000000.0)
    }, index=pd.date_range("2021-01-01", periods=400))
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:350], params)
    pd.testing.assert_frame_equal(full.iloc[:350], prefix)
    assert strategy.generate_signals(sample, full, None, params).data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
