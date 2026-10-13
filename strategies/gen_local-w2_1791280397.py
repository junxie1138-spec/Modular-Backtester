from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 12
    regime_window: int = 10
    atr_window: int = 14
    innovation_threshold: float = 0.025
    regime_threshold: float = 0.45
    breakeven_pct: float = 0.004
    initial_stop_atr: float = 1.2
    trail_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791280397"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(params.trend_window, params.atr_window) + params.regime_window + 2)

    def indicators(self, data, params):
        if min(params.trend_window, params.regime_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (params.innovation_threshold > 0 and 0 < params.regime_threshold < 1):
            raise ValueError("Invalid innovation or regime threshold.")
        if not (params.breakeven_pct > 0 and params.initial_stop_atr > 0 and params.trail_atr > 0):
            raise ValueError("Stop parameters must be positive.")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be one or two sessions.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["noise_scale"] = (out["atr"].shift(1) / out["previous_close"]).where(
            lambda x: x > 0
        )
        n = params.trend_window
        # Finite, causal local-linear latent trend; convolution avoids Python rolling callbacks.
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        slopes = np.full(len(data), np.nan)
        if len(data) >= n:
            slopes[n - 1:] = np.convolve(
                out["log_close"].to_numpy(), weights[::-1], mode="valid"
            ) / np.dot(weights, weights)
        out["slope"] = slopes
        out["level"] = out["log_close"].rolling(n).mean() + out["slope"] * (n - 1) / 2.0
        out["forecast"] = out["level"].shift(1) + out["slope"].shift(1)
        out["forecast_error"] = (out["log_close"] - out["forecast"]) / out["noise_scale"]
        out["slope_innovation"] = out["slope"].diff() / out["noise_scale"]
        out["error_sum"] = out["forecast_error"].rolling(params.regime_window).sum()
        out["absolute_error_sum"] = out["forecast_error"].abs().rolling(params.regime_window).sum()
        out["persistence"] = out["error_sum"] / out["absolute_error_sum"].where(
            out["absolute_error_sum"] > 0
        )
        out["plastic"] = out["persistence"] > params.regime_threshold
        out["elastic"] = out["persistence"].abs() <= params.regime_threshold
        out["recovering_error"] = out["forecast_error"] > out["forecast_error"].shift(1)
        out["entry"] = (
            (out["slope"] > 0)
            & (out["slope_innovation"] > params.innovation_threshold)
            & ((out["plastic"] & (out["forecast_error"] > 0))
               | (out["elastic"] & (out["forecast_error"] < 0) & out["recovering_error"]))
        ).fillna(False)
        out["strength"] = (
            out["slope_innovation"] / (3.0 * params.innovation_threshold)
        ).clip(0.25, 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.25)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        raw_signal = np.zeros(len(data), dtype=int)
        raw_size = np.full(len(data), 0.25, dtype=float)
        active = False
        pending = False
        armed = False
        entry = stop = entry_atr = 0.0
        position_size = 0.25
        bars_held = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if active:
                opening = float(data["open"].iloc[i])
                low = float(data["low"].iloc[i])
                closing = float(data["close"].iloc[i])
                atr = float(indicators["atr"].iloc[i])
                if pending:
                    # The prior close ordered entry; this bar's open is now observable.
                    entry = opening
                    stop = entry - params.initial_stop_atr * entry_atr
                    pending = False
                bars_held += 1
                valid = np.isfinite([opening, low, closing, atr, entry]).all() and entry > 0 and atr > 0
                exit_now = not valid or low <= stop
                if not exit_now:
                    # Close-confirmed activation avoids guessing the intraday high/low ordering.
                    armed = armed or closing >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, closing - params.trail_atr * atr)
                    exit_now = closing <= stop or bars_held >= params.max_hold
                if exit_now:
                    active = False
                    armed = False
                else:
                    raw_signal[i] = 1
                    raw_size[i] = position_size
                # Never reopen on the same closing decision as an exit.
                continue
            atr = float(indicators["atr"].iloc[i])
            if i >= warmup and bool(indicators["entry"].iloc[i]) and np.isfinite(atr) and atr > 0:
                position_size = float(indicators["strength"].iloc[i])
                if not np.isfinite(position_size):
                    continue
                active = pending = True
                armed = False
                bars_held = 0
                entry_atr = atr
                raw_signal[i] = 1
                raw_size[i] = position_size
        df["signal"] = raw_signal
        df["size"] = raw_size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    w = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=w + 5, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.5, "close": 101.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"atr": 2.0, "entry": False, "strength": 0.7}, index=index)
    ind.loc[index[w], "entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=w)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist()[w:w + 4] == [0, 1, 1, 0]
    assert (result.loc[index[w + 1:w + 3], "size"] == 0.7).all()
    data.loc[index[w + 1], "low"] = 97.0
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist()[w:w + 4] == [0, 1, 0, 0]
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.generate_signals(data, actual, ctx, params).data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
