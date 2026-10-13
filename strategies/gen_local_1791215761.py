from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    depth_atr: float = 2.0
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1791215761"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    @staticmethod
    def warmup_bars(params: GapRepairParams) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        if not all(np.isfinite(x) and x > 0 for x in (params.depth_atr, params.trail_atr)):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_peak"] = data["close"].rolling(20, min_periods=20).max().shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        # [pandas rolling documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["atr"] = out["atr"].where(out["atr"] > 0)
        out["previous_atr"] = out["atr"].shift(1)
        out["drawdown"] = out["previous_peak"] - out["previous_close"]
        out["depth_atr"] = out["drawdown"] / out["previous_atr"]
        out["opening_gap"] = data["open"] - out["previous_close"]
        out["repair_fraction"] = out["opening_gap"] / out["drawdown"].where(out["drawdown"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        valid &= (data[["open", "high", "low", "close"]] > 0).all(axis=1)
        valid &= out.notna().all(axis=1)
        out["entry"] = (
            valid
            & (out["depth_atr"] >= params.depth_atr)
            & (out["repair_fraction"] >= 1.0 / 3.0)
            & (out["repair_fraction"] < 1.0)
            & (data["close"] >= data["open"])
            & (data["close"] < out["previous_peak"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        highest_close = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                # The previous decision fills at today's open; only in-trade closes enter this maximum.
                held += 1
                if valid:
                    highest_close = max(highest_close, close[i])
                    stop = max(stop, highest_close - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or held >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif valid and entry[i]:
                active = True
                held = 0
                highest_close = -np.inf
                stop = -np.inf
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 98.0,
         "close": [100.0, 101.0, 99.0, 100.0, 101.0, 102.0], "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, GapRepairParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    warmup = GeneratedStrategy.indicators(data, GapRepairParams())
    assert not warmup["entry"].any()
    missing = ind.copy()
    missing.loc[index[1], "atr"] = np.nan
    result = GeneratedStrategy.generate_signals(data, missing, None, GapRepairParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
