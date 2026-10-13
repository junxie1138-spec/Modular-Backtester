from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalTensionParams:
    history_bars: int = 504
    pullback_bars: int = 3
    min_samples: int = 30
    seasonal_floor: float = 0.002
    pullback_threshold: float = 0.006
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalTensionParams]):
    strategy_id = "gen_local_1790934818"

    @classmethod
    def params_type(cls):
        return SeasonalTensionParams

    def warmup_bars(self, params):
        return int(max(params.history_bars + params.holding_bars,
                       params.pullback_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        if (params.history_bars < 1 or params.pullback_bars < 1
                or params.min_samples < 1 or params.atr_bars < 1
                or not 3 <= params.holding_bars <= 5
                or params.seasonal_floor < 0 or params.pullback_threshold <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        # Each outcome is available only after its entire holding window ends.
        outcome = close.pct_change(params.holding_bars, fill_method=None)
        origin_weekday = weekday.shift(params.holding_bars)
        seasonal = pd.Series(np.nan, index=data.index, dtype=float)
        samples = pd.Series(0.0, index=data.index)
        out = pd.DataFrame(index=data.index)
        out["holding_return"] = outcome
        out["origin_weekday"] = origin_weekday
        for day in range(5):
            valid = origin_weekday.eq(day) & np.isfinite(outcome)
            count = valid.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars).sum()
            total = outcome.where(valid, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars).sum()
            mean = total.div(count.where(count > 0))
            out[f"weekday_{day}_count"] = count
            out[f"weekday_{day}_mean"] = mean
            seasonal = seasonal.where(~weekday.eq(day), mean)
            samples = samples.where(~weekday.eq(day), count)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        out["seasonal_mean"] = seasonal
        out["seasonal_samples"] = samples
        out["pullback_return"] = close.pct_change(
            params.pullback_bars, fill_method=None)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        seasonal_agrees = (
            indicators["seasonal_mean"].gt(params.seasonal_floor)
            & indicators["seasonal_samples"].ge(params.min_samples)
        )
        pullback_agrees = indicators["pullback_return"].lt(
            -params.pullback_threshold)
        ready = (seasonal_agrees & pullback_agrees
                 & indicators["atr"].gt(0)).fillna(False).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        active = False
        entry = np.nan
        entry_atr = np.nan
        stop = np.nan
        armed = False
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if active:
                # The previous close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opening
                    stop = entry - params.trail_atr * entry_atr
                age += 1
                usable = (np.isfinite([opening, high, low, close, entry, stop]).all()
                          and entry > 0 and np.isfinite(atr[i]) and atr[i] > 0)
                exit_now = not usable
                if usable:
                    # Test the existing stop before raising it with this bar.
                    exit_now = low <= stop
                    if not exit_now:
                        armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                        if armed:
                            stop = max(stop, entry, high - params.trail_atr * atr[i])
                            exit_now = close <= stop
                    exit_now = exit_now or age >= params.holding_bars
                if exit_now:
                    active = False
                    entry = np.nan
                    armed = False
                    age = 0
                else:
                    raw[i] = 1
            elif (i >= self.warmup_bars(params) - 1 and ready[i]
                  and np.isfinite(close) and close > 0):
                active = True
                entry = np.nan
                entry_atr = atr[i]
                armed = False
                age = 0
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = SeasonalTensionParams(history_bars=1, pullback_bars=1,
                              min_samples=1, atr_bars=1)
    idx = pd.bdate_range("2020-01-01", periods=12)
    bars = pd.DataFrame({"open": 100.0, "high": 100.4, "low": 99.6,
                         "close": 100.0, "volume": 1000.0}, index=idx)
    ind = pd.DataFrame({"seasonal_mean": 0.01, "seasonal_samples": 1.0,
                        "pullback_return": 0.0, "atr": 1.0}, index=idx)
    ind.loc[idx[4], "pullback_return"] = -0.02
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    signals = strategy.generate_signals(bars, ind, ctx, p).data
    assert signals["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    # Arm at +0.8%; a larger ATR next day must never lower the stop.
    bars.loc[idx[5], ["high", "low", "close"]] = [102.0, 99.9, 101.8]
    bars.loc[idx[6], ["open", "high", "low", "close"]] = [101.8, 102.0, 100.1, 101.0]
    ind.loc[idx[6], "atr"] = 5.0
    signals = strategy.generate_signals(bars, ind, ctx, p).data
    assert signals["signal"].iloc[5:8].tolist() == [1, 1, 0]
    # Either missing primitive prevents entry.
    ind.loc[idx[4], "seasonal_mean"] = np.nan
    assert strategy.generate_signals(bars, ind, ctx, p).data["signal"].eq(0).all()
    assert strategy.indicators(bars, p).index.equals(bars.index)
