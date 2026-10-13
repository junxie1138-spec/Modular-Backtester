from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnLaneParams:
    return_window: int = 20
    pressure_window: int = 3
    pressure_threshold: float = 0.15
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[ReturnLaneParams]):
    strategy_id = "gen_local_1790916671"

    @classmethod
    def params_type(cls):
        return ReturnLaneParams

    def warmup_bars(self, params: ReturnLaneParams) -> int:
        return int(max(params.return_window + 2, params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ReturnLaneParams) -> pd.DataFrame:
        if min(params.return_window, params.pressure_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if not np.isfinite(params.pressure_threshold) or not 0 <= params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must be in [0, 1).")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        history = out["previous_return"].rolling(params.return_window, min_periods=params.return_window)
        out["return_lower"] = history.quantile(0.2)
        out["return_upper"] = history.quantile(0.8)
        out["return_median"] = history.median()
        out["bar_range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) /
                                 out["bar_range"].replace(0, np.nan)).clip(-1, 1)
        out["pressure"] = out["close_location"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["bar_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ReturnLaneParams,
    ) -> SignalFrame:
        # Primitive 1: a return escapes the opposite-polarity historical lane.
        long_return = ((indicators["return"] > indicators["return_upper"]) &
                       (indicators["return"] > 0) & (indicators["return_median"] < 0))
        short_return = ((indicators["return"] < indicators["return_lower"]) &
                        (indicators["return"] < 0) & (indicators["return_median"] > 0))
        # Primitive 2: recent within-bar closing pressure must independently agree.
        long_entry = long_return & (indicators["pressure"] > params.pressure_threshold)
        short_entry = short_return & (indicators["pressure"] < -params.pressure_threshold)
        required = ["return", "return_lower", "return_upper", "return_median", "pressure", "atr"]
        ready = np.isfinite(indicators[required]).all(axis=1) & (indicators["atr"] > 0)
        entries = np.select([long_entry & ready, short_entry & ready], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan
        for i in range(len(data)):
            if position:
                # The order decided yesterday fills at today's open; no future row is read.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i]):
                    position = 0
                elif (position * (closes[i] - entry_price) < -stop_distance or
                      i - entry_bar + 1 >= params.holding_bars):
                    position = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame({
        "open": 100.0, "high": 120.0, "low": 80.0,
        "close": [120, 110, 99, 95, 100, 80, 90, 101, 105, 100, 100],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "return": 0.0, "return_lower": -0.01, "return_upper": 0.01,
        "return_median": 0.0, "pressure": 0.0, "atr": 2.0,
    }, index=index)
    ind.loc[index[0], ["return", "return_median", "pressure"]] = [0.02, -0.001, 0.5]
    ind.loc[index[5], ["return", "return_median", "pressure"]] = [-0.02, 0.001, -0.5]
    ind.loc[index[1:], "atr"] = 20.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    strategy = GeneratedStrategy()
    params = ReturnLaneParams()
    # Keep the short's entry ATR at two; intervening ATR changes must not move stops.
    ind.loc[index[5], "atr"] = 2.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0]
    params.holding_bars = 2
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
