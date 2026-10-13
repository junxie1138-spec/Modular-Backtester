from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    echo_window: int = 30
    echo_min: float = 0.15
    volume_window: int = 20
    pressure_min: float = 0.35
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    max_hold: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791331002"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(params.echo_window + 4, params.volume_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if min(params.echo_window, params.volume_window, params.atr_window) < 3:
            raise ValueError("Lookbacks must be at least three bars.")
        if not (0.0 < params.echo_min < 1.0 and params.pressure_min > 0.0):
            raise ValueError("Invalid entry thresholds.")
        if not (params.breakeven_pct > 0.0 and params.trail_atr > 0.0):
            raise ValueError("Stop parameters must be positive.")
        if params.max_hold not in (1, 2) or not (0.0 < params.size <= 1.0):
            raise ValueError("Holding horizon must be one or two bars; size must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["roc"] = data["close"].pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_lag1"] = out["acceleration"].shift(1)
        out["acceleration_lag2"] = out["acceleration"].shift(2)
        rolling = out["acceleration"].rolling(params.echo_window, min_periods=params.echo_window)
        out["corr_lag1"] = rolling.corr(out["acceleration_lag1"])
        out["corr_lag2"] = rolling.corr(out["acceleration_lag2"])
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / out["range"].replace(0.0, np.nan)
        )
        out["volume_mean"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out["pressure"] = (
            out["close_location"] * data["volume"]
            / out["volume_mean"].replace(0.0, np.nan)
        )
        out["previous_pressure"] = out["pressure"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        # Primitive 1: positive acceleration recurs in an alternating, two-day echo.
        out["acceleration_agrees"] = (
            (out["acceleration"] > 0.0)
            & (out["acceleration_lag1"] < 0.0)
            & (out["corr_lag1"] < -params.echo_min)
            & (out["corr_lag2"] > params.echo_min)
        ).fillna(False)
        # Primitive 2: volume-weighted close location flips from selling to buying.
        out["pressure_agrees"] = (
            (out["previous_pressure"] <= 0.0)
            & (out["pressure"] > params.pressure_min)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: EchoParams,
    ) -> SignalFrame:
        eligible = (
            indicators["acceleration_agrees"].fillna(False)
            & indicators["pressure_agrees"].fillna(False)
            & (indicators["atr"] > 0.0)
        ).to_numpy(dtype=bool)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = peak = stop = np.nan
        held = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = bool(np.isfinite(prices[i]).all() and np.all(prices[i] > 0.0))
            if pending:
                # The previous close requested entry; this bar's open is the reference fill.
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    entry = opening
                    peak = opening
                    stop = entry - params.trail_atr * atr[i - 1]
                    active = True
                    armed = False
                    held = 0
            if active:
                held += 1
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    active = False
                    continue
                # Test the stop carried into the bar before raising it using this bar's high.
                touched = low <= stop
                peak = max(peak, high)
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Newly raised stops also trigger on a close breach.
                # Every exit is a close decision, executed at the following open.
                if touched or close <= stop or held >= params.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif valid and eligible[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EchoParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": [100.0] * 6,
         "high": [100.0, 100.6, 101.0, 100.0, 102.0, 100.0],
         "low": [100.0, 99.7, 100.1, 100.0, 99.8, 100.0],
         "close": [100.0, 100.4, 100.7, 100.0, 100.2, 100.0],
         "volume": [1000.0] * 6}, index=index
    )
    indicators = pd.DataFrame(
        {"acceleration_agrees": [True, False, False, True, False, False],
         "pressure_agrees": [True, False, False, True, False, False],
         "atr": [1.0] * 6}, index=index
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 0]
    # A close above breakeven exits against the raised trail, before the time limit.
    assert data["close"].iloc[4] > data["open"].iloc[4]
    indicators.loc[index[0], "pressure_agrees"] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:4].eq(0).all()
    assert result["size"].gt(0.0).all()
    flat = data.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    calculated = strategy.indicators(flat, params)
    assert not strategy.generate_signals(flat, calculated, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
