from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    visibility_window: int = 12
    breakout_window: int = 12
    atr_window: int = 14
    hub_fraction: float = 0.60
    stop_atr: float = 1.50
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790857672"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.visibility_window + 1, params.breakout_window + 1, params.atr_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if min(params.visibility_window, params.breakout_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0 < params.hub_fraction <= 1 or not params.stop_atr > 0:
            raise ValueError("Invalid topology threshold or stop multiplier")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("Holding period must be three to five sessions")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_close"] = np.log(close.where(close > 0))
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["channel_high"] = data["high"].rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.breakout_window, min_periods=params.breakout_window).min().shift(1)

        n = len(data)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        minimum = np.full(n, np.inf)
        maximum = np.full(n, -np.inf)
        for lag in range(1, params.visibility_window + 1):
            slope = ((out["log_close"] - out["log_close"].shift(lag)) / lag).to_numpy()
            finite = np.isfinite(slope)
            # An upper edge exists when every nearer slope is strictly larger.
            upper += finite & (slope < minimum)
            lower += finite & (slope > maximum)
            minimum = np.fmin(minimum, slope)
            maximum = np.fmax(maximum, slope)

        complete = out["log_close"].rolling(params.visibility_window + 1, min_periods=params.visibility_window + 1).count().eq(params.visibility_window + 1)
        out["upper_density"] = pd.Series(upper / params.visibility_window, index=data.index).where(complete)
        out["lower_density"] = pd.Series(lower / params.visibility_window, index=data.index).where(complete)
        ready = out[["atr", "channel_high", "channel_low", "upper_density", "lower_density"]].notna().all(axis=1) & out["atr"].gt(0)
        out["long_setup"] = ready & close.gt(out["channel_high"]) & out["upper_density"].ge(params.hub_fraction) & out["upper_density"].gt(out["lower_density"])
        out["short_setup"] = ready & close.lt(out["channel_low"]) & out["lower_density"].ge(params.hub_fraction) & out["lower_density"].gt(out["upper_density"])
        out["long_confirmed"] = out["long_setup"] & out["long_setup"].shift(1, fill_value=False)
        out["short_confirmed"] = out["short_setup"] & out["short_setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        position = 0
        age = 0
        stop = np.nan

        for i in range(len(data)):
            # Yesterday's decision enters at today's open; freeze yesterday's ATR.
            if position == 0 and i > 0 and targets[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(targets[i - 1])
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    age = 0

            if position != 0:
                age += 1
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) <= 0
                if stopped or age >= params.hold_bars:
                    position = 0
                    # This close schedules liquidation at the next open.
                    targets[i] = 0
                    continue
                targets[i] = position
            elif np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i]:
                    targets[i] = 1
                elif shorts[i]:
                    targets[i] = -1

        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "long_confirmed": False, "short_confirmed": False}, index=index)
    ind.loc[index[2], "long_confirmed"] = True
    data.loc[index[4], "close"] = 98.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    ind["long_confirmed"] = False
    ind.loc[index[2], "short_confirmed"] = True
    data.loc[index[4], "close"] = 102.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, -1, -1, 0, 0, 0]
    t = np.arange(40, dtype=float)
    prices = np.exp(4.0 + 0.001 * t * t)
    sample = pd.DataFrame({"open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices, "volume": 1000.0}, index=pd.date_range("2021-01-01", periods=40))
    topology = strategy.indicators(sample, params)
    assert topology["upper_density"].iloc[-1] == 1.0
    assert np.isclose(topology["lower_density"].iloc[-1], 1.0 / params.visibility_window)
    assert topology["long_confirmed"].iloc[-1]
    prefix = strategy.indicators(sample.iloc[:30], params)
    pd.testing.assert_frame_equal(prefix, topology.iloc[:30])


if __name__ == "__main__":
    _self_check()
