from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorParams:
    floor_window: int = 3
    drift_window: int = 3
    atr_window: int = 5
    percentile_window: int = 126
    entry_quantile: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[FloorParams]):
    strategy_id = "gen_local_1790916188"

    @classmethod
    def params_type(cls) -> type[FloorParams]:
        return FloorParams

    @staticmethod
    def warmup_bars(params: FloorParams) -> int:
        return int(max(
            params.atr_window + 1,
            params.floor_window + params.drift_window + params.percentile_window,
            params.drift_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FloorParams) -> pd.DataFrame:
        if min(params.floor_window, params.drift_window, params.atr_window, params.percentile_window) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["floor"] = data["low"].rolling(params.floor_window).min()
        out["floor_lift"] = out["floor"].diff(params.drift_window)
        out["strength"] = out["floor_lift"] / out["atr"].where(out["atr"] > 0)
        out["strength"] = out["strength"].replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["strength"].shift(1).rolling(params.percentile_window).quantile(params.entry_quantile)
        out["close_change"] = data["close"].diff(params.drift_window)
        out["entry"] = (
            out["strength"].notna()
            & out["threshold"].notna()
            & (out["strength"] > out["threshold"])
            & (out["strength"] > 0)
            & (out["close_change"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FloorParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                signal[i] = 0
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    for holding, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
    ):
        result = GeneratedStrategy.generate_signals(data, indicators, None, FloorParams(hold_bars=holding))
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()
