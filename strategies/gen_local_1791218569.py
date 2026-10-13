from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_window: int = 20
    atr_window: int = 14
    min_streak: int = 3
    compression_ratio: float = 0.85
    confirmation_location: float = 0.65
    stop_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791218569"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.range_window + 3, params.atr_window + 1, params.min_streak + 3))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if params.range_window < 1 or params.atr_window < 1 or params.min_streak < 1:
            raise ValueError("Lookbacks and min_streak must be positive")
        if not 0 < params.compression_ratio <= 1 or not 0 <= params.confirmation_location <= 1:
            raise ValueError("Invalid compression or confirmation threshold")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        valid &= (data[["open", "high", "low", "close"]] > 0).all(axis=1)
        valid &= data["high"] >= data[["open", "close"]].max(axis=1)
        valid &= data["low"] <= data[["open", "close"]].min(axis=1)
        out["valid"] = valid
        out["range"] = (data["high"] - data["low"]).where(valid)
        out["lower_wick"] = (data[["open", "close"]].min(axis=1) - data["low"]).where(valid)
        previous_close = data["close"].where(valid).shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(valid & previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["range_baseline"] = out["range"].rolling(params.range_window, min_periods=params.range_window).median().shift(1)
        out["contracting"] = (
            (out["range"] > 0)
            & (out["range"] < out["range"].shift(1))
            & (out["lower_wick"] < out["lower_wick"].shift(1))
        ).fillna(False)
        out["streak"] = out["contracting"].astype(int).groupby((~out["contracting"]).cumsum()).cumsum()
        out["setup"] = (
            (out["streak"] >= params.min_streak)
            & (out["range"] <= params.compression_ratio * out["range_baseline"])
        ).fillna(False)
        out["close_location"] = ((data["close"] - data["low"]) / out["range"].replace(0, np.nan)).where(valid)
        out["confirmation"] = (
            valid
            & valid.shift(1, fill_value=False)
            & (data["close"] > data["close"].shift(1))
            & (out["close_location"] >= params.confirmation_location)
        ).fillna(False)
        out["entry"] = (
            out["setup"].shift(2, fill_value=False)
            & out["confirmation"].shift(1, fill_value=False)
            & out["confirmation"]
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        pending = active = False
        stop = frozen_atr = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                active = np.isfinite(opens[i]) and opens[i] > 0
                # Entry executes at this bar's open; ATR was frozen at the preceding decision close.
                stop = opens[i] - params.stop_atr * frozen_atr
                age = 0
            if active:
                age += 1
                if not np.isfinite(close[i]) or close[i] < stop or age >= 2:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                frozen_atr = atr[i]
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "close": [100.0, 98.9, 100.0, 100.0, 100.0]}, index=index)
    features = pd.DataFrame({"atr": [1.0, 100.0, 100.0, 100.0, 100.0], "entry": [True, False, False, False, False]}, index=index)
    stopped = strategy.generate_signals(data, features, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0, 0]
    data["close"] = 100.0
    expired = strategy.generate_signals(data, features, None, params).data
    assert expired["signal"].tolist() == [0, 1, 1, 0, 0]
    data["open"] = [100.0, 110.0, 100.0, 100.0, 100.0]
    assert strategy.generate_signals(data, features, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    for n in range(1, 6):
        prefix = strategy.generate_signals(data.iloc[:n], features.iloc[:n], None, params).data
        assert prefix.equals(strategy.generate_signals(data, features, None, params).data.iloc[:n])


if __name__ == "__main__":
    _self_check()
