from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CorridorParams:
    range_bars: int = 20
    atr_bars: int = 14
    corridor_atr: float = 1.25
    expansion: float = 1.04
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CorridorParams]):
    strategy_id = "gen_local-w2_1791337762"

    @classmethod
    def params_type(cls) -> type[CorridorParams]:
        return CorridorParams

    def warmup_bars(self, params: CorridorParams) -> int:
        return int(max(200, params.range_bars + 1, params.atr_bars + 2))

    def indicators(self, data: pd.DataFrame, params: CorridorParams) -> pd.DataFrame:
        if (params.range_bars < 2 or params.atr_bars < 2
                or not 3 <= params.hold_bars <= 5
                or not np.isfinite(params.corridor_atr) or params.corridor_atr <= 0
                or not np.isfinite(params.expansion) or params.expansion <= 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid corridor parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["previous_atr"] = out["atr"].shift(1).where(lambda x: x > 0)
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["midpoint"] = (out["range_high"] + out["range_low"]) / 2.0
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["displacement"] = close - out["midpoint"]
        out["old_distance"] = out["displacement"] / out["previous_atr"]
        out["new_distance"] = out["displacement"] / out["atr"].where(lambda x: x > 0)
        out["atr_expansion"] = out["atr"] / out["previous_atr"]
        out["close_movement"] = (close - out["previous_close"]).abs() / out["previous_atr"]
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        quiet_expansion = ready & (out["atr_expansion"] >= params.expansion) & (out["close_movement"] <= 0.35)
        out["long_entry"] = (
            quiet_expansion & (close > out["ma200"])
            & (out["old_distance"] < -params.corridor_atr)
            & (out["new_distance"] >= -params.corridor_atr)
            & (out["new_distance"] < 0)
        )
        out["short_entry"] = (
            quiet_expansion & (close < out["ma200"])
            & (out["old_distance"] > params.corridor_atr)
            & (out["new_distance"] <= params.corridor_atr)
            & (out["new_distance"] > 0)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CorridorParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        fill_bar = -1
        entry_atr = 0.0
        fixed_stop = np.nan
        for i in range(len(data)):
            if position:
                if i == fill_bar:
                    # The previous close ordered this fill; ATR is frozen there.
                    fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = position * (closes[i] - fixed_stop) < 0
                expired = i - fill_bar + 1 >= params.hold_bars
                invalid = not np.isfinite(closes[i]) or not np.isfinite(fixed_stop)
                if stopped or expired or invalid:
                    position = 0
                    continue
                signals[i] = position
            elif np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                position = 1 if longs[i] else (-1 if shorts[i] else 0)
                if position:
                    fill_bar = i + 1
                    entry_atr = atr[i]
                    signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="B")
    close = np.array([100, 110, 99, 95, 100, 110, 100, 115, 110], dtype=float)
    opening = np.array([100, 100, 110, 99, 95, 110, 110, 100, 115], dtype=float)
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 1,
        "low": np.minimum(opening, close) - 1, "close": close, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 2.0,
        "long_entry": [True, False, False, False, False, False, False, False, False],
        "short_entry": [False, False, False, False, False, True, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, CorridorParams()).data
    # Both stops use fill opens and fixed entry ATR; favorable moves do not ratchet them.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    expiry = strategy.generate_signals(
        data, indicators, None, CorridorParams(stop_atr=100.0, hold_bars=3)
    ).data
    assert expiry["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1]
    empty_entries = indicators.copy()
    empty_entries["atr"] = np.nan
    assert not strategy.generate_signals(data, empty_entries, None, CorridorParams()).data["signal"].any()
    assert strategy.warmup_bars(CorridorParams()) >= 200


if __name__ == "__main__":
    _self_check()
