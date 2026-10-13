from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 20
    compression_ceiling: float = 0.95
    trail_atr: float = 2.5
    atr_window: int = 14
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791256508"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.range_window + 1, params.atr_window + 1, 5))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["relative_range"] = (data["high"] - data["low"]) / previous_close.replace(0.0, np.nan)
        out["range_fast"] = out["relative_range"].rolling(3, min_periods=3).mean()
        out["range_baseline"] = out["relative_range"].shift(1).rolling(params.range_window, min_periods=params.range_window).mean()
        out["range_ratio"] = out["range_fast"] / out["range_baseline"].replace(0.0, np.nan)
        out["range_acceleration"] = out["range_fast"].diff()
        out["close_location"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0.0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["seasonal_window"] = (data.index.day >= 7) & (data.index.day <= 15)
        out["entry_setup"] = (
            out["seasonal_window"]
            & out["range_ratio"].gt(0.50)
            & out["range_ratio"].le(params.compression_ceiling)
            & out["range_acceleration"].gt(0.0)
            & out["close_location"].ge(0.65)
            & data["close"].gt(previous_close)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        setup = indicators["entry_setup"].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                held += 1
                if not valid:
                    active = False
                    continue
                # Entry fills on this bar after the previous bar's decision.
                high_water = close if not np.isfinite(high_water) else max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or held >= params.holding_bars:
                    active = False
                    continue
                signals[i] = 1
            elif confirmed[i] and valid:
                active = True
                held = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    closes = np.array([100.0, 100.0, 100.0, 104.0, 103.0, 99.0, 99.0, 99.0])
    data = pd.DataFrame({"open": closes, "high": closes + 1.0, "low": closes - 1.0, "close": closes, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_setup": [True, True, False, False, False, False, False, False], "atr": [1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params(trail_atr=2.0))
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0.0).all()
    indicators["entry_setup"] = [True, False, False, False, False, False, False, False]
    assert strategy.generate_signals(data, indicators, None, Params()).data["signal"].eq(0).all()
    derived = strategy.indicators(data, Params())
    assert derived.index.equals(data.index)
    assert not derived["entry_setup"].any()


if __name__ == "__main__":
    _self_check()
