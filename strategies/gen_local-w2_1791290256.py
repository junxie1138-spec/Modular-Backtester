from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    momentum_bars: int = 3
    percentile_bars: int = 126
    acceleration_quantile: float = 0.85
    atr_bars: int = 14
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791290256"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.momentum_bars + params.percentile_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.momentum_bars < 1 or params.percentile_bars < 2 or params.atr_bars < 1:
            raise ValueError("Lookbacks must be positive; percentile_bars must be at least two.")
        if not 0.0 < params.acceleration_quantile < 1.0:
            raise ValueError("acceleration_quantile must lie strictly between zero and one.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("trail_atr must be positive and finite; hold_bars must be three to five.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        opening = data["open"].where(data["open"] > 0.0)
        previous = close.shift(1)
        out["roc"] = close.div(close.shift(params.momentum_bars)).sub(1.0)
        out["acceleration"] = out["roc"].diff()
        out["intraday_return"] = np.log(close.div(opening))
        out["overnight_return"] = np.log(opening.div(previous))
        out["intraday_momentum"] = np.expm1(out["intraday_return"].rolling(params.momentum_bars).sum())
        out["overnight_momentum"] = np.expm1(out["overnight_return"].rolling(params.momentum_bars).sum())
        out["intraday_acceleration"] = out["intraday_momentum"].diff()
        out["overnight_acceleration"] = out["overnight_momentum"].diff()
        out["acceleration_threshold"] = out["acceleration"].shift(1).rolling(params.percentile_bars).quantile(params.acceleration_quantile)
        ranges = pd.concat([
            data["high"].sub(data["low"]),
            data["high"].sub(previous).abs(),
            data["low"].sub(previous).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & out["roc"].gt(0.0)
            & out["acceleration"].gt(0.0)
            & out["acceleration"].gt(out["acceleration_threshold"])
            & out["intraday_acceleration"].gt(0.0)
            & out["overnight_acceleration"].le(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            price = closes[i]
            atr = atrs[i]
            valid = np.isfinite(price) and price > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or price <= stop or age >= params.hold_bars:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                # Entry executes next bar; start tracking closes on that bar.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100.0, 101.0, 104.0, 103.0, 102.0, 105.0, 106.0, 107.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0], "entry": [True, False, False, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params(trail_atr=2.0, hold_bars=5))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data["close"] = np.arange(100.0, 108.0)
    indicators["atr"] = 1.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params(hold_bars=4))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
