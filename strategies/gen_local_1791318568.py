from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDiscountParams:
    ma_window: int = 20
    atr_window: int = 14
    discount_z: float = 0.6
    max_discount_z: float = 2.2
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalDiscountParams]):
    strategy_id = "gen_local_1791318568"

    @classmethod
    def params_type(cls):
        return SeasonalDiscountParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.ma_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data, params):
        if params.ma_window < 2 or params.atr_window < 1:
            raise ValueError("Indicator windows must be positive; ma_window must be at least two.")
        if not (0 < params.discount_z < params.max_discount_z):
            raise ValueError("Discount thresholds must be positive and ordered.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Exit parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["weekday"] = data.index.dayofweek
        out["discount"] = (
            out["z"].between(-params.max_discount_z, -params.discount_z)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
            & np.isfinite(close)
        ).fillna(False)
        out["confirmed"] = (
            out["weekday"].eq(1)
            & out["weekday"].shift(1).eq(0)
            & out["discount"]
            & out["discount"].shift(1, fill_value=False)
            & out["z"].gt(out["z"].shift(1))
            & close.gt(out["previous_close"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                previous_atr = atrs[i - 1]
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(previous_atr) and previous_atr > 0:
                    entry = opens[i]
                    stop = entry - params.trail_atr * previous_atr
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                price = closes[i]
                # Stops are evaluated at the close and execute at the next open.
                if not np.isfinite(price) or price <= stop or age >= params.hold_bars:
                    active = False
                    continue
                if price >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, price - params.trail_atr * atrs[i])
                signals[i] = 1
            elif entries[i]:
                # The entry decision is filled at the following bar's open.
                signals[i] = 1
                pending = True
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-08", periods=7)
    close = np.array([99.0, 99.5, 101.2, 103.0, 102.5, 101.8, 101.0])
    data = pd.DataFrame({
        "open": [99.0, 99.0, 100.0, 101.0, 103.0, 102.0, 101.5],
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 0.5, 4.0, 4.0, 4.0],
        "confirmed": [False, True, False, False, False, False, False],
    }, index=index)
    params = SeasonalDiscountParams(hold_bars=5)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    # Entry uses Wednesday's open; increased ATR cannot lower Thursday's stop.
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    params.hold_bars = 3
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
