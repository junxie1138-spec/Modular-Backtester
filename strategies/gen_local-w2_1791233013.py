from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReleaseParams:
    volume_bars: int = 20
    atr_bars: int = 14
    stall_atr: float = 0.15
    release_atr: float = 0.35
    stall_volume: float = 1.3
    release_volume: float = 1.1
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReleaseParams]):
    strategy_id = "gen_local-w2_1791233013"

    @classmethod
    def params_type(cls) -> type[ReleaseParams]:
        return ReleaseParams

    def warmup_bars(self, params: ReleaseParams) -> int:
        return max(200, params.volume_bars + 2, params.atr_bars + 2)

    def indicators(self, data: pd.DataFrame, params: ReleaseParams) -> pd.DataFrame:
        if min(params.volume_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if min(params.stall_atr, params.release_atr, params.stall_volume,
               params.release_volume, params.trail_atr) <= 0:
            raise ValueError("Thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["move"] = close.diff()
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["volume_base"] = data["volume"].rolling(
            params.volume_bars, min_periods=params.volume_bars
        ).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_base"].where(
            out["volume_base"] > 0
        )
        out["close_position"] = (close - data["low"]) / out["range"].where(
            out["range"] > 0
        )
        out["entry"] = (
            (close > out["ma200"])
            & (out["move"].shift(1).abs() <= params.stall_atr * out["atr"].shift(1))
            & (out["relative_volume"].shift(1) >= params.stall_volume)
            & (out["move"] >= params.release_atr * out["atr"])
            & (out["relative_volume"] >= params.release_volume)
            & (out["relative_volume"] < out["relative_volume"].shift(1))
            & (out["close_position"] >= 0.75)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ReleaseParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest = stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    # Only closes after the next-bar entry enter the high-water mark.
                    highest = max(highest, close[i])
                    stop = max(stop, highest - params.trail_atr * atr[i])
                if not valid or close[i] < stop or age >= params.hold_bars:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                highest = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, ReleaseParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0]
    data["close"] = [100.0, 103.0, 102.0, 102.0]
    ind["atr"] = [1.0, 1.0, 10.0, 10.0]
    params = ReleaseParams(trail_atr=0.5, hold_bars=10)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
