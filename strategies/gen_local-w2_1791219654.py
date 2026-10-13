from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    shape_bars: int = 6
    library_bars: int = 180
    neighbors: int = 5
    holding_bars: int = 8
    atr_bars: int = 14
    stop_atr: float = 2.5
    minimum_edge: float = 0.4
    maximum_distance: float = 0.9


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791219654"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.library_bars + max(
            params.fractional_window + params.shape_bars, params.atr_bars + 1
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if not (0.0 < p.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if min(p.fractional_window, p.shape_bars, p.neighbors,
               p.holding_bars, p.atr_bars) < 2:
            raise ValueError("Window lengths and neighbors must be at least two")
        first_lag = p.holding_bars + p.shape_bars
        if p.library_bars - first_lag + 1 < p.neighbors:
            raise ValueError("Analog library is too short")
        if min(p.stop_atr, p.minimum_edge, p.maximum_distance) <= 0:
            raise ValueError("Thresholds must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.ones(p.fractional_window)
        for j in range(1, p.fractional_window):
            weights[j] = -weights[j - 1] * (p.fractional_order - j + 1) / j
        fractional = close * weights[0]
        for j in range(1, p.fractional_window):
            fractional = fractional + weights[j] * close.shift(j)
        out["fractional_close"] = fractional
        out["shape_mean"] = fractional.rolling(p.shape_bars).mean()
        out["shape_std"] = fractional.rolling(p.shape_bars).std(ddof=0)
        scale = out["shape_std"].where(out["shape_std"] > 1e-12)
        shape_columns = []
        for j in range(p.shape_bars):
            name = f"shape_{j}"
            shape_columns.append(name)
            out[name] = (fractional.shift(j) - out["shape_mean"]) / scale

        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(p.atr_bars).mean()
        shape = out[shape_columns]
        distances, outcomes = [], []
        # ponytail: bounded O(bars * library * shape); chunk the library if it grows.
        for lag in range(first_lag, p.library_bars + 1):
            distance = np.sqrt(
                (shape - shape.shift(lag)).pow(2).mean(axis=1, skipna=False)
            )
            outcome = (
                close.shift(lag - p.holding_bars) - close.shift(lag)
            ) / out["atr"].shift(lag).replace(0.0, np.nan)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(distance.where(valid, np.inf).to_numpy())
            outcomes.append(outcome.where(valid, 0.0).to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argsort(distance_matrix, axis=1)[:, :p.neighbors]
        selected_distance = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcome_matrix, nearest, axis=1)
        weight = np.exp(-selected_distance)
        denominator = weight.sum(axis=1)
        edge = np.divide(
            (weight * selected_outcome).sum(axis=1), denominator,
            out=np.zeros(len(data)), where=denominator > 0,
        )
        wins = np.divide(
            (weight * (selected_outcome > 0)).sum(axis=1), denominator,
            out=np.zeros(len(data)), where=denominator > 0,
        )
        complete = np.isfinite(selected_distance).all(axis=1)
        ready = complete & (np.arange(len(data)) >= self.warmup_bars(p) - 1)
        out["expected_atr_gain"] = np.where(ready, edge, 0.0)
        out["win_fraction"] = np.where(ready, wins, 0.0)
        out["analog_distance"] = np.where(
            ready, selected_distance.mean(axis=1), np.nan
        )
        out["conviction"] = (
            (out["expected_atr_gain"] / (2.0 * p.minimum_edge)).clip(0.0, 1.0)
            * out["win_fraction"]
            * np.exp(-out["analog_distance"])
        ).fillna(0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams,
    ) -> SignalFrame:
        p = params
        enter = (
            (indicators["expected_atr_gain"] >= p.minimum_edge)
            & (indicators["win_fraction"] >= 0.6)
            & (indicators["analog_distance"] <= p.maximum_distance)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        conviction = indicators["conviction"].fillna(0.0).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.2, dtype=float)
        active, age, high, stop, entry_size = False, 0, np.nan, -np.inf, 0.2
        for i in range(len(data)):
            if active:
                age += 1
                if np.isfinite(close[i]):
                    high = close[i] if not np.isfinite(high) else max(high, close[i])
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, high - p.stop_atr * atr[i])
                if age >= p.holding_bars or (
                    np.isfinite(close[i]) and close[i] <= stop
                ):
                    active = False
                else:
                    signal[i], size[i] = 1, entry_size
            elif enter[i] and np.isfinite(close[i]):
                active, age, high, stop = True, 0, np.nan, -np.inf
                entry_size = float(0.2 + 0.75 * np.clip(conviction[i], 0.0, 1.0))
                signal[i], size[i] = 1, entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, p = GeneratedStrategy(), AnalogParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=650, freq="B")
    rng = np.random.default_rng(17)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 650))), index=index)
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    indicators = strategy.indicators(data, p)
    full = strategy.generate_signals(data, indicators, ctx, p).data
    prefix = data.iloc[:450]
    prefix_indicators = strategy.indicators(prefix, p)
    pd.testing.assert_frame_equal(indicators.iloc[:450], prefix_indicators)
    pd.testing.assert_frame_equal(
        full.iloc[:450], strategy.generate_signals(prefix, prefix_indicators, ctx, p).data
    )
    assert full["signal"].isin([0, 1]).all()
    assert full["size"].between(0.2, 0.95).all()
    probe = data.iloc[:5].copy()
    probe["close"] = [100.0, 105.0, 104.0, 101.0, 100.0]
    features = pd.DataFrame({
        "expected_atr_gain": [1.0, 0.0, 0.0, 0.0, 0.0],
        "win_fraction": 1.0, "analog_distance": 0.0,
        "conviction": 0.8, "atr": [1.0, 1.0, 3.0, 3.0, 3.0],
    }, index=probe.index)
    result = strategy.generate_signals(probe, features, ctx, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.8)


if __name__ == "__main__":
    _self_check()
