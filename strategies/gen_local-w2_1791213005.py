from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_window: int = 40
    fd_order: float = 0.4
    z_window: int = 60
    regime_window: int = 40
    regime_threshold: float = 0.05
    entry_z: float = 0.8
    pressure_window: int = 10
    pressure_threshold: float = 0.08
    atr_window: int = 14
    trail_atr: float = 3.5
    hold_bars: int = 18
    spike_atr: float = 2.5
    cooldown_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791213005"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.fd_window + params.z_window - 1,
            params.regime_window + 2,
            params.atr_window + params.cooldown_bars + 1,
            params.pressure_window,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        windows = (p.fd_window, p.z_window, p.regime_window,
                   p.pressure_window, p.atr_window, p.cooldown_bars)
        if any(w < 2 for w in windows) or p.hold_bars < 1:
            raise ValueError("Windows must be >= 2; holding period must be positive.")
        numeric = (p.fd_order, p.regime_threshold, p.entry_z,
                   p.pressure_threshold, p.trail_atr, p.spike_atr)
        if not all(np.isfinite(v) for v in numeric):
            raise ValueError("Parameters must be finite.")
        if not (0 < p.fd_order < 1 and 0 < p.regime_threshold < 1
                and p.entry_z > 0 and 0 < p.pressure_threshold < 1
                and p.trail_atr > 0 and p.spike_atr > 0):
            raise ValueError("Invalid signal or stop parameters.")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        result = pd.DataFrame(index=data.index)
        weights = np.ones(p.fd_window, dtype=float)
        for lag in range(1, p.fd_window):
            weights[lag] = -weights[lag - 1] * (p.fd_order - lag + 1) / lag
        # ponytail: fixed finite-memory filter; increase fd_window for longer memory.
        fd = close.copy()
        for lag in range(1, p.fd_window):
            fd = fd + weights[lag] * close.shift(lag)
        result["fractional_close"] = fd
        mean = fd.rolling(p.z_window, min_periods=p.z_window).mean()
        std = fd.rolling(p.z_window, min_periods=p.z_window).std(ddof=0)
        result["fd_mean"] = mean
        result["fd_std"] = std
        result["fd_z"] = (fd - mean) / std.where(std > 0)
        returns = close.pct_change(fill_method=None)
        result["returns"] = returns
        result["regime_corr"] = returns.rolling(
            p.regime_window, min_periods=p.regime_window
        ).corr(returns.shift(1))
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        volume = data["volume"].astype(float).clip(lower=0)
        body = ((close - data["open"]) / span).clip(-1, 1)
        result["candle_pressure"] = body
        result["pressure"] = (body * volume).rolling(
            p.pressure_window, min_periods=p.pressure_window
        ).sum() / volume.rolling(
            p.pressure_window, min_periods=p.pressure_window
        ).sum().replace(0, np.nan)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        result["true_range"] = tr
        atr = tr.rolling(p.atr_window, min_periods=p.atr_window).mean()
        result["atr"] = atr.where(atr > 0)
        spikes = (tr > p.spike_atr * atr.shift(1)).astype(float)
        spikes = spikes.where(atr.shift(1).notna())
        result["spike"] = spikes
        result["blocked"] = spikes.rolling(
            p.cooldown_bars, min_periods=p.cooldown_bars
        ).max()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        p = params
        z = indicators["fd_z"]
        corr = indicators["regime_corr"]
        pressure = indicators["pressure"]
        # Persistent returns follow fractional displacement; alternating returns fade it.
        primary = pd.Series(0, index=data.index, dtype=int)
        trend = corr >= p.regime_threshold
        reverting = corr <= -p.regime_threshold
        primary.loc[(trend & (z >= p.entry_z)) |
                    (reverting & (z <= -p.entry_z))] = 1
        primary.loc[(trend & (z <= -p.entry_z)) |
                    (reverting & (z >= p.entry_z))] = -1
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[(primary == 1) & (pressure >= p.pressure_threshold)] = 1
        candidate.loc[(primary == -1) & (pressure <= -p.pressure_threshold)] = -1
        required = indicators[["fd_z", "regime_corr", "pressure", "atr", "blocked"]]
        valid = np.isfinite(required.to_numpy(dtype=float)).all(axis=1)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = candidate.to_numpy(dtype=int)
        blocked = indicators["blocked"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = low_water = 0.0
        floor = -np.inf
        short_ceiling = np.inf
        for i in range(len(data)):
            close = closes[i]
            if not np.isfinite(close) or close <= 0 or not valid[i]:
                position = 0
                continue
            if position:
                high_water = max(high_water, close)
                floor = max(floor, high_water - p.trail_atr * atrs[i])
                low_water = min(low_water, close)
                if position == -1:
                    short_ceiling = min(
                        short_ceiling, low_water + p.trail_atr * atrs[i]
                    )
                # The high-water floor applies to both directions.
                # For shorts it realizes a downward move; the ceiling limits adverse moves.
                stopped = close <= floor or (position == -1 and close >= short_ceiling)
                if stopped or i - entry_bar >= p.hold_bars:
                    position = 0
                    continue
            elif blocked[i] == 0 and candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
                high_water = low_water = close
                floor = close - p.trail_atr * atrs[i]
                short_ceiling = close + p.trail_atr * atrs[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    p = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 105., 110., 108., 106., 107.]}, index=index)
    features = pd.DataFrame({
        "fd_z": 2., "regime_corr": 0.5, "pressure": 0.5,
        "atr": 1., "blocked": 0.,
    }, index=index)
    output = strategy.generate_signals(data, features, None, p).data
    assert output["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data["close"] = [100., 99., 96., 96., 96., 96.]
    features["fd_z"] = -2.
    features["pressure"] = -0.5
    output = strategy.generate_signals(data, features, None, p).data
    assert output["signal"].iloc[:4].tolist() == [0, -1, -1, 0]
    features["pressure"] = 0.5
    assert (strategy.generate_signals(data, features, None, p).data["signal"] == 0).all()
    assert strategy.warmup_bars(p) >= p.fd_window + p.z_window - 1


if __name__ == "__main__":
    _self_check()
