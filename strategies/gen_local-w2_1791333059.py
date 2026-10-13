from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRenewalParams:
    window: int = 10
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[RangeRenewalParams]):
    strategy_id = "gen_local-w2_1791333059"

    @classmethod
    def params_type(cls) -> type[RangeRenewalParams]:
        return RangeRenewalParams

    @staticmethod
    def warmup_bars(params: RangeRenewalParams) -> int:
        return max(15, params.window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeRenewalParams) -> pd.DataFrame:
        if params.window < 2 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("window must be >= 2 and stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        out["range"] = (high - low).where(high > low)
        out["previous_high"] = high.shift(1)
        out["previous_low"] = low.shift(1)
        out["previous_close"] = close.shift(1)
        upper_boundary = pd.concat([low, out["previous_high"]], axis=1).max(axis=1, skipna=False)
        lower_boundary = pd.concat([high, out["previous_low"]], axis=1).min(axis=1, skipna=False)
        out["upward_new"] = (high - upper_boundary).clip(lower=0)
        out["downward_new"] = (lower_boundary - low).clip(lower=0)
        overlap_high = pd.concat([high, out["previous_high"]], axis=1).min(axis=1, skipna=False)
        overlap_low = pd.concat([low, out["previous_low"]], axis=1).max(axis=1, skipna=False)
        out["overlap"] = (overlap_high - overlap_low).clip(lower=0) / out["range"]
        out["renewal"] = (out["upward_new"] - out["downward_new"]) / out["range"]
        out["strength"] = out["renewal"].rolling(params.window, min_periods=params.window).mean()
        out["mean_overlap"] = out["overlap"].rolling(params.window, min_periods=params.window).mean()
        out["previous_strength"] = out["strength"].shift(1)
        out["true_range"] = pd.concat([
            high - low,
            (high - out["previous_close"]).abs(),
            (low - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["setup"] = (
            (out["strength"] > 0.12)
            & (out["previous_strength"] <= 0.12)
            & (out["mean_overlap"] >= 0.55)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeRenewalParams) -> SignalFrame:
        wanted = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setup = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        holding = pending = False
        stop = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                # Entry fills at today's open; volatility was known at yesterday's close.
                stop = opens[i] - params.stop_atr * atr[i - 1]
                pending = False
            if holding:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(stop) or close[i] < stop or age >= 8:
                    holding = False
            elif setup[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                holding = pending = True
                age = 0
            wanted[i] = int(holding)
        df = pd.DataFrame({"signal": wanted, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeRenewalParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "setup": [True] + [False] * 11}, index=index)
    def signals(frame: pd.DataFrame, derived: pd.DataFrame) -> pd.Series:
        return GeneratedStrategy.generate_signals(frame, derived, None, params).data["signal"]
    assert signals(data, ind).tolist() == [0] + [1] * 8 + [0] * 3
    data.loc[index[3], "close"] = 97.0
    ind.loc[index[2]:, "atr"] = 50.0
    assert signals(data, ind).tolist() == [0, 1, 1, 1] + [0] * 8
    data.loc[index[1], "open"] = 110.0
    data.loc[index[1], "close"] = 109.0
    data.loc[index[2], "close"] = 107.0
    assert signals(data, ind).tolist() == [0, 1, 1] + [0] * 9
    for n in range(1, len(data) + 1):
        assert signals(data.iloc[:n], ind.iloc[:n]).equals(signals(data, ind).iloc[:n])


if __name__ == "__main__":
    _self_check()
