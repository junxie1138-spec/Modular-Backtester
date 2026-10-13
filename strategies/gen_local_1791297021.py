from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    analog_lookback: int = 252
    analog_horizon: int = 5
    volume_multiple: float = 1.15
    min_forecast: float = 0.004
    min_matches: int = 5
    breakeven_pct: float = 0.02
    atr_multiple: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791297021"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @classmethod
    def warmup_bars(cls, params: AnalogParams) -> int:
        return int(params.analog_lookback + params.analog_horizon + 21)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change()
        volume_mean = data["volume"].rolling(20, min_periods=20).mean().replace(0, np.nan)
        volume_ok = (data["volume"] / volume_mean >= params.volume_multiple).fillna(False)
        up = returns.gt(0).astype(int)
        shape = up + 2 * up.shift(1, fill_value=0) + 4 * up.shift(2, fill_value=0)
        valid = returns.shift(2).notna() & volume_ok
        past_shape = shape.shift(params.analog_horizon)
        past_valid = valid.shift(params.analog_horizon).fillna(False)
        outcome = close.pct_change(params.analog_horizon).replace([np.inf, -np.inf], np.nan)
        forecast = pd.Series(0.0, index=data.index)
        support = pd.Series(0.0, index=data.index)
        for code in range(8):
            match = past_valid & past_shape.eq(code)
            count = match.astype(int).rolling(params.analog_lookback, min_periods=params.analog_lookback).sum()
            total = outcome.where(match, 0.0).rolling(params.analog_lookback, min_periods=params.analog_lookback).sum()
            forecast = forecast.where(shape.ne(code), total.div(count.where(count.gt(0))))
            support = support.where(shape.ne(code), count)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        return pd.DataFrame({
            "forecast": forecast.fillna(0.0),
            "support": support.fillna(0.0),
            "volume_ok": volume_ok,
            "atr": atr.fillna(0.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": 1.0}, index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        forecasts = indicators["forecast"].to_numpy()
        supports = indicators["support"].to_numpy()
        volumes = indicators["volume_ok"].to_numpy()
        atrs = indicators["atr"].to_numpy()
        position = 0
        entry = stop = 0.0
        armed = False
        bars = 0
        for i in range(len(data)):
            if i and raw[i - 1] != position:
                position = raw[i - 1]
                if position:
                    entry = float(opens[i])
                    stop = entry - position * params.atr_multiple * atrs[i - 1]
                    armed = False
                    bars = 0
            if position:
                bars += 1
                hit = lows[i] <= stop if position == 1 else highs[i] >= stop
                if hit or bars >= params.max_hold:
                    raw[i] = 0
                    continue
                if position == 1:
                    if highs[i] >= entry * (1 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, highs[i] - params.atr_multiple * atrs[i])
                else:
                    if lows[i] <= entry * (1 - params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = min(stop, entry, lows[i] + params.atr_multiple * atrs[i])
                raw[i] = position
            elif i >= self.warmup_bars(params) and atrs[i] > 0 and volumes[i] and volumes[i - 1] and supports[i] >= params.min_matches and supports[i - 1] >= params.min_matches:
                if forecasts[i] > params.min_forecast and forecasts[i - 1] > params.min_forecast and closes[i] > closes[i - 1] > closes[i - 2]:
                    raw[i] = 1
                elif forecasts[i] < -params.min_forecast and forecasts[i - 1] < -params.min_forecast and closes[i] < closes[i - 1] < closes[i - 2]:
                    raw[i] = -1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
