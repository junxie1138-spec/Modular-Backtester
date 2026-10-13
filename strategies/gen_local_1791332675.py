from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_period: int = 3
    mean_period: int = 20
    atr_period: int = 14
    min_negative_roc: float = 0.006
    recovery_fraction: float = 0.15
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791332675"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.mean_period, params.atr_period + 1, params.roc_period + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["roc"] = close.pct_change(periods=params.roc_period, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        remaining_decline = (-out["roc"].shift(1)).where(out["roc"].shift(1) < 0)
        out["recovery_fraction"] = (out["acceleration"] / remaining_decline).replace([np.inf, -np.inf], np.nan)
        out["mean"] = close.rolling(params.mean_period, min_periods=params.mean_period).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["setup"] = (
            (out["roc"] < -params.min_negative_roc)
            & (out["acceleration"] > 0)
            & (out["recovery_fraction"] >= params.recovery_fraction)
            & (close < out["mean"])
            & np.isfinite(close)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.stop_atr <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("stop_atr must be positive and hold_bars must be 1 or 2")
        setup = indicators["setup"].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop_price = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                    active = True
                    age = 0
            if active:
                age += 1
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                if stopped or age >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    sample = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    sample.iloc[3, sample.columns.get_loc("close")] = 98.0
    derived = pd.DataFrame({"setup": [False, True, True, False, False, True, True, False, False, False], "atr": 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(sample, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
