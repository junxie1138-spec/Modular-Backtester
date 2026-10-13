from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    frac_window: int = 24
    norm_window: int = 40
    archive_bars: int = 160
    atr_window: int = 14
    hold_bars: int = 4
    frac_d: float = 0.45
    bandwidth: float = 0.75
    edge_atr: float = 0.4
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791278047"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        periods = (params.frac_window, params.norm_window, params.archive_bars, params.atr_window)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in periods):
            raise ValueError("Lookbacks must be integers >= 2.")
        if not isinstance(params.hold_bars, int) or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer from 3 to 5.")
        values = (params.frac_d, params.bandwidth, params.edge_atr, params.trail_atr)
        if not all(np.isfinite(v) and v > 0 for v in values) or params.frac_d >= 1:
            raise ValueError("Require 0 < frac_d < 1 and positive finite scales.")
        return int(max(params.atr_window + 1, params.frac_window + params.norm_window + 4 + params.archive_bars + params.hold_bars))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        clean = data[["high", "low", "close"]].astype(float).replace([np.inf, -np.inf], np.nan)
        close = clean["close"].where(clean["close"] > 0)
        previous = close.shift(1)
        tr = pd.concat([clean["high"] - clean["low"], (clean["high"] - previous).abs(), (clean["low"] - previous).abs()], axis=1).max(axis=1, skipna=False)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr = atr.where(atr > 0)
        log_close = np.log(close)
        fractional = log_close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; extend frac_window if truncation matters.
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_d) / lag
            fractional = fractional + weight * log_close.shift(lag)
        mean = fractional.rolling(params.norm_window, min_periods=params.norm_window).mean()
        scale = fractional.rolling(params.norm_window, min_periods=params.norm_window).std(ddof=0)
        z = (fractional - mean) / scale.where(scale > 0)
        weighted = pd.Series(0.0, index=data.index)
        mass = weighted.copy()
        squared_mass = weighted.copy()
        matches = weighted.copy()
        # Four-point fractional-close levels preserve displacement and curvature.
        for lag in range(params.hold_bars + 1, params.hold_bars + params.archive_bars + 1):
            distance = (z - z.shift(lag)).pow(2)
            for offset in range(1, 4):
                distance = distance + (z.shift(offset) - z.shift(lag + offset)).pow(2)
            outcome = (close.shift(lag - params.hold_bars) - close.shift(lag)) / atr.shift(lag)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            kernel = np.exp(-distance / (8.0 * params.bandwidth ** 2)).where(valid, 0.0)
            weighted = weighted + kernel * outcome.where(valid, 0.0)
            mass = mass + kernel
            squared_mass = squared_mass + kernel.pow(2)
            matches = matches + valid.astype(float)
        effective = mass.pow(2) / squared_mass.where(squared_mass > 0)
        forecast = (weighted / mass.where(mass > 0)).where((matches >= 8) & (effective >= 8))
        return pd.DataFrame({"atr": atr, "fractional_close": fractional, "fractional_mean": mean, "fractional_scale": scale, "fractional_z": z, "analog_mass": mass, "analog_matches": matches, "effective_matches": effective, "forecast_atr": forecast}, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        self.warmup_bars(params)
        forecast = indicators["forecast_atr"]
        raw = pd.Series(np.select([forecast.ge(params.edge_atr), forecast.le(-params.edge_atr)], [1, -1], default=0), index=data.index)
        confirmed = raw.ne(0) & raw.eq(raw.shift(1).fillna(0))
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        directions = raw.to_numpy(dtype=int)
        entries = confirmed.to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        highest, stop = np.nan, -np.inf
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position, age = 0, 0
                highest, stop = np.nan, -np.inf
                continue
            if position:
                age += 1
                highest = close[i] if np.isnan(highest) else max(highest, close[i])
                stop = max(stop, highest - params.trail_atr * atr[i])
                # The requested highest-close barrier applies to both directions.
                if close[i] <= stop or age >= params.hold_bars:
                    position = 0
                signal[i] = position
                continue
            if entries[i]:
                position, age = int(directions[i]), 0
                # First held bar is the next bar, after the mandatory signal shift.
                highest, stop = np.nan, -np.inf
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), AnalogParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": [100., 100., 103., 100., 100., 100., 100., 101., 98., 98.]}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "forecast_atr": [1., 1., 1., 1., 0., -1., -1., -1., 0., 0.]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, -1, -1, 0]
    indicators["forecast_atr"] = 1.0
    indicators["atr"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:7].tolist() == [0, 0, 1, 1, 1, 1, 0]
    n = strategy.warmup_bars(params) + 40
    x = np.arange(n, dtype=float)
    close = 100.0 + 0.03 * x + 2.0 * np.sin(x / 7.0)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=n))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-8], params)
    assert np.allclose(full.iloc[:-8].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert np.isfinite(full["forecast_atr"].iloc[-1])
    result = strategy.generate_signals(data, full, ctx, params).data
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
