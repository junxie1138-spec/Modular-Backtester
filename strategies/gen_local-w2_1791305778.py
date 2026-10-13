from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fd_order: float = 0.45
    fd_window: int = 40
    ar_window: int = 60
    forecast_threshold: float = 0.20
    atr_window: int = 14
    breakeven_pct: float = 0.018
    initial_atr: float = 2.5
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791305778"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.fd_window + params.ar_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (0.0 < params.fd_order < 1.0):
            raise ValueError("fd_order must be between zero and one")
        if min(params.fd_window, params.ar_window, params.atr_window) < 3:
            raise ValueError("indicator windows must be at least three")
        if params.forecast_threshold <= 0 or params.max_hold_bars < 1:
            raise ValueError("forecast_threshold and max_hold_bars must be positive")
        if min(params.breakeven_pct, params.initial_atr, params.trail_atr) <= 0:
            raise ValueError("stop parameters must be positive")

        log_close = np.log(data["close"].where(data["close"] > 0))
        weights = [1.0]
        for lag in range(1, params.fd_window):
            weights.append(-weights[-1] * (params.fd_order - lag + 1) / lag)
        fractional_close = log_close * weights[0]
        for lag, weight in enumerate(weights[1:], start=1):
            fractional_close = fractional_close + weight * log_close.shift(lag)

        increment = fractional_close.diff()
        lag_one = increment.shift(1)
        lag_two = increment.shift(2)
        window = params.ar_window
        mean_y = increment.rolling(window).mean()
        mean_one = lag_one.rolling(window).mean()
        mean_two = lag_two.rolling(window).mean()
        var_one = lag_one.rolling(window).var()
        var_two = lag_two.rolling(window).var()
        cov_one_two = lag_one.rolling(window).cov(lag_two)
        cov_y_one = increment.rolling(window).cov(lag_one)
        cov_y_two = increment.rolling(window).cov(lag_two)
        determinant = var_one * var_two - cov_one_two.pow(2)
        determinant = determinant.where(determinant > 1e-8 * var_one * var_two)
        beta_one = (cov_y_one * var_two - cov_y_two * cov_one_two) / determinant
        beta_two = (cov_y_two * var_one - cov_y_one * cov_one_two) / determinant
        forecast = mean_y + beta_one * (increment - mean_one) + beta_two * (lag_one - mean_two)
        scale = increment.rolling(window).std().replace(0.0, np.nan)
        strength = forecast / scale

        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        ready = (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        ready = ready & strength.notna() & beta_one.notna() & beta_two.notna() & atr.gt(0)
        result = pd.DataFrame({
            "fractional_close": fractional_close,
            "increment": increment,
            "beta_one": beta_one,
            "beta_two": beta_two,
            "forecast": forecast,
            "strength": strength,
            "atr": atr,
            "ready": ready.astype(int),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        setup = (
            indicators["ready"].eq(1)
            & indicators["increment"].lt(0)
            & indicators["beta_one"].lt(-0.05)
            & indicators["beta_two"].gt(0.15)
            & indicators["strength"].gt(params.forecast_threshold)
        ).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_bar = 0
        position_size = 0.20

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.initial_atr * atr[i - 1]
                    armed = False
                    active = True

            if active:
                sizes[i] = position_size
                if lows[i] <= stop or i - entry_bar + 1 >= params.max_hold_bars:
                    active = False
                    continue
                if highs[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    # Raises apply to subsequent bars; no assumed intrabar high/low ordering.
                    stop = max(stop, entry, highs[i] - params.trail_atr * atr[i])
                signal[i] = 1
            elif setup[i]:
                position_size = float(np.clip(strength[i] / 1.5, 0.20, 0.95))
                sizes[i] = position_size
                signal[i] = 1
                pending = True

        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(max_hold_bars=10)
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.5, 103.0, 102.0, 101.0, 100.5, 100.5],
        "low": [99.5, 99.5, 99.8, 100.2, 99.5, 99.5],
        "close": [100.0, 102.0, 100.0, 100.5, 100.0, 100.0],
        "volume": [1000000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "ready": [1, 0, 0, 0, 0, 0],
        "increment": [-0.01] * 6,
        "beta_one": [-0.2] * 6,
        "beta_two": [0.4] * 6,
        "strength": [0.6] * 6,
        "atr": [1.0] * 6,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.4)
    assert result["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert np.isfinite(derived.to_numpy(dtype=float)).all()


if __name__ == "__main__":
    self_check()
