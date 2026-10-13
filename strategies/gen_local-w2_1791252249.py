from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_window: int = 20
    min_drawdown: float = 0.012
    max_drawdown: float = 0.06
    atr_window: int = 14
    breakeven_gain: float = 0.008
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791252249"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.drawdown_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.drawdown_window >= 2 and params.atr_window >= 1
                and 0 < params.min_drawdown < params.max_drawdown < 1
                and params.breakeven_gain > 0 and params.trail_atr > 0
                and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["peak"] = data["close"].rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["drawdown_change"] = out["drawdown"].diff()
        calendar = data.index.isocalendar()
        out["week_key"] = (calendar["year"] * 100 + calendar["week"]).to_numpy(dtype=np.int64)
        out["first_session"] = out["week_key"].ne(out["week_key"].shift(1))
        ready = np.isfinite(out[["atr", "ma200", "drawdown", "drawdown_change"]]).all(axis=1)
        out["entry"] = (ready & out["first_session"]
                        & (out["atr"] > 0)
                        & (data["close"] > out["ma200"])
                        & out["drawdown"].between(params.min_drawdown, params.max_drawdown)
                        & (out["drawdown_change"] > 0)).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry = np.nan
        stop = -np.inf
        watermark = -np.inf
        armed = False
        age = 0
        for i in range(n):
            opening, high, low, close = prices[i]
            valid = (np.isfinite(prices[i]).all() and np.isfinite(atr[i])
                     and np.isfinite(ma[i]) and opening > 0 and low > 0
                     and high >= low and atr[i] > 0)
            if active:
                if not valid:
                    active = False
                    continue
                # The previous close's entry decision fills at this open.
                if np.isnan(entry):
                    entry = opening
                    stop = entry - params.trail_atr * atr[i - 1]
                    watermark = entry
                age += 1
                # Test the previously established stop before raising it.
                # Every exit is a close decision filled at the following open.
                if low <= stop or age >= params.hold_bars or close <= ma[i]:
                    active = False
                else:
                    watermark = max(watermark, high)
                    armed = armed or high >= entry * (1.0 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry, watermark - params.trail_atr * atr[i])
                    if close <= stop:
                        active = False
                raw[i] = int(active)
            elif valid and entries[i]:
                active = True
                entry = np.nan
                stop = -np.inf
                watermark = -np.inf
                armed = False
                age = 0
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": np.ones(n, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-06", periods=6, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.0, 101.2, 100.8, 100.6, 100.0, 100.0],
        "low": [99.0, 99.8, 100.1, 99.9, 99.0, 99.0],
        "close": [100.0, 101.0, 100.5, 100.2, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
        "ma200": [90.0] * 6,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    # ATR expansion must not lower the breakeven stop; exits shift one bar.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    data["high"] = 100.2
    data["low"] = 99.8
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
