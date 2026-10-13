from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CeilingTestParams:
    range_bars: int = 15
    streak_bars: int = 4
    ceiling_floor: float = 0.80
    max_upper_wick: float = 0.20
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CeilingTestParams]):
    strategy_id = "gen_local-w2_1791242875"

    @classmethod
    def params_type(cls):
        return CeilingTestParams

    @staticmethod
    def warmup_bars(params: CeilingTestParams) -> int:
        return int(params.range_bars + params.streak_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CeilingTestParams) -> pd.DataFrame:
        if params.range_bars < 2 or params.streak_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.ceiling_floor < 1.0:
            raise ValueError("ceiling_floor must be between zero and one")
        if not 0.0 <= params.max_upper_wick <= 1.0:
            raise ValueError("max_upper_wick must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["range_width"] = out["range_high"] - out["range_low"]
        out["relative_position"] = ((data["close"] - out["range_low"]) / out["range_width"].where(out["range_width"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["ceiling_test"] = (out["relative_position"].ge(params.ceiling_floor) & out["relative_position"].le(1.0)).fillna(False)
        out["ceiling_streak"] = out["ceiling_test"].astype(int).groupby((~out["ceiling_test"]).cumsum()).cumsum()
        out["candle_range"] = data["high"] - data["low"]
        out["upper_wick_fraction"] = ((data["high"] - data[["open", "close"]].max(axis=1)) / out["candle_range"].where(out["candle_range"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["confirmation"] = (out["ceiling_test"] & out["upper_wick_fraction"].ge(0.0) & out["upper_wick_fraction"].le(params.max_upper_wick)).fillna(False)
        out["entry_ready"] = (out["ceiling_streak"].ge(params.streak_bars) & out["confirmation"] & out["confirmation"].shift(1, fill_value=False)).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CeilingTestParams) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be three, four, or five")
        ready = indicators["entry_ready"].fillna(False).astype(bool)
        entries = (ready & ~ready.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force an exit bar even if a fresh setup appears here.
                    exit_at = -1
            elif enter:
                exit_at = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = CeilingTestParams(range_bars=2)
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"open": 8.0, "high": 10.0, "low": 0.0, "close": 9.0, "volume": 100.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert derived["ceiling_streak"].iloc[:6].tolist() == [0, 0, 1, 2, 3, 4]
    assert not derived["entry_ready"].iloc[:5].any()
    assert derived["entry_ready"].iloc[5]
    ready = pd.DataFrame({"entry_ready": [False, True, False, True, False, False, False, True, False, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, ready, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
