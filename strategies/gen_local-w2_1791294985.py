from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    range_bars: int = 20
    roc_bars: int = 3
    acceleration_min: float = 0.001
    atr_bars: int = 14
    trail_atr: float = 0.75
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791294985"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.range_bars + 1, params.roc_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.range_bars, params.roc_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if not np.isfinite(params.acceleration_min) or params.acceleration_min < 0:
            raise ValueError("acceleration_min must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min()
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max()
        out["range_width"] = out["range_high"] - out["range_low"]
        out["position"] = ((close - out["range_low"]) / out["range_width"].where(out["range_width"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_position"] = out["position"].shift(1)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["ready"] = np.isfinite(out[["ma200", "position", "previous_position", "roc", "acceleration", "atr"]]).all(axis=1) & (out["atr"] > 0) & np.isfinite(close)
        out["regime"] = (close > out["ma200"]) & out["ready"]
        out["entry"] = (
            out["regime"]
            & (out["previous_position"] <= 0.5)
            & (out["position"] > 0.5)
            & (out["roc"] > 0)
            & (out["acceleration"] > params.acceleration_min)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        regimes = indicators["regime"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                if not valid or not regimes[i]:
                    active = False
                    continue
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or i - entry_bar >= params.hold_bars:
                    active = False
                    continue
            elif valid and regimes[i] and entries[i]:
                active = True
                entry_bar = i
                high_water = close
                stop = high_water - params.trail_atr * atr
            signals[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 104.0, 102.0, 105.0, 106.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0] * 5, "entry": [True, False, False, False, False], "regime": [True] * 5}, index=index)
    params = StrategyParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = [100.0, 104.0, 103.0, 105.0, 106.0]
    indicators["atr"] = [1.0, 1.0, 10.0, 1.0, 1.0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
