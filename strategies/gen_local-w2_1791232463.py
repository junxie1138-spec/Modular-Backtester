from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    level_gain: float = 0.35
    slope_gain: float = 0.08
    innovation_threshold: float = 0.08
    atr_period: int = 14
    trail_atr: float = 1.6
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791232463"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (0 < params.slope_gain <= params.level_gain <= 1):
            raise ValueError("Require 0 < slope_gain <= level_gain <= 1")
        if not (params.atr_period >= 1 and 3 <= params.hold_days <= 5):
            raise ValueError("Require atr_period >= 1 and hold_days in [3, 5]")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0
                and np.isfinite(params.innovation_threshold)
                and params.innovation_threshold >= 0):
            raise ValueError("Require finite positive trail_atr and nonnegative threshold")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()

        # Causal alpha-beta state observer: level predicts price, slope predicts drift.
        n = len(data)
        levels = np.full(n, np.nan)
        slopes = np.full(n, np.nan)
        predictions = np.full(n, np.nan)
        slope_updates = np.full(n, np.nan)
        level = np.nan
        slope = 0.0
        for i, price in enumerate(close.to_numpy()):
            if not np.isfinite(price) or price <= 0:
                level = np.nan
                slope = 0.0
                continue
            if not np.isfinite(level):
                level = price
                levels[i] = level
                slopes[i] = slope
                continue
            prediction = level + slope
            residual = price - prediction
            update = params.slope_gain * residual
            level = prediction + params.level_gain * residual
            slope += update
            predictions[i] = prediction
            levels[i] = level
            slopes[i] = slope
            slope_updates[i] = update

        out["predicted_level"] = predictions
        out["latent_level"] = levels
        out["latent_slope"] = slopes
        out["slope_update"] = slope_updates
        out["previous_slope"] = out["latent_slope"].shift(1)
        out["slope_innovation"] = out["slope_update"].div(
            out["atr"].where(out["atr"] > 0)
        )
        out["bull_regime"] = (close > out["ma200"]).fillna(False)
        out["entry"] = (
            out["bull_regime"]
            & (out["previous_slope"] <= 0)
            & (out["latent_slope"] > 0)
            & (out["slope_innovation"] >= params.innovation_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["bull_regime"].fillna(False).to_numpy(dtype=bool)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest_close = -np.inf
        stop = -np.inf
        for i, price in enumerate(prices):
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # Entry fills on this bar's open; only in-trade closes set the peak.
                    highest_close = max(highest_close, price)
                    stop = max(stop, highest_close - params.trail_atr * atr[i])
                    if price <= stop or age >= params.hold_days or not regime[i]:
                        active = False
            elif valid and regime[i] and entries[i]:
                active = True
                age = 0
                highest_close = -np.inf
                stop = -np.inf
            raw[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(hold_days=5)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": [100., 110., 103., 104., 103., 102., 102., 102., 102.]}, index=index)
    ind = pd.DataFrame({
        "atr": [1., 1., 1., 1., 10., 10., 1., 1., 1.],
        "bull_regime": True,
        "entry": [False, True, False, False, False, False, False, False, False],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, params).data
    # A pre-entry close of 110 cannot set the peak; ATR expansion cannot lower the stop.
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    ind["atr"] = 1.0
    data["close"] = 100.0
    params.hold_days = 4
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=260)
    close = pd.Series(100 + np.arange(260) * 0.1 + np.sin(np.arange(260)), index=index)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(bars, params)
    prefix = strategy.indicators(bars.iloc[:230], params)
    pd.testing.assert_frame_equal(full.iloc[:230], prefix)
    assert not full["entry"].iloc[:199].any()
    signals = strategy.generate_signals(bars, full, ctx, params).data
    assert set(signals["signal"].unique()) <= {0, 1}
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
