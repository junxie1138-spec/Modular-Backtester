from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    ma_window: int = 20
    compression_window: int = 5
    catchup_bars: int = 3
    compression_ratio: float = 0.45
    z_floor: float = -1.5
    z_ceiling: float = -0.25
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local_1790888123"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        return int(max(200, params.ma_window + params.catchup_bars, params.compression_window))

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        if not (2 <= params.compression_window < params.ma_window and params.catchup_bars >= 1):
            raise ValueError("Require 2 <= compression_window < ma_window and catchup_bars >= 1")
        if not (0.0 < params.compression_ratio < 1.0 and params.z_floor < params.z_ceiling < 0.0):
            raise ValueError("Require a compression ratio between zero and one and ordered negative z thresholds")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"] / out["std"].where(out["std"] > 0.0)
        out["fast_span"] = (
            data["high"].rolling(params.compression_window, min_periods=params.compression_window).max()
            - data["low"].rolling(params.compression_window, min_periods=params.compression_window).min()
        )
        out["slow_span"] = (
            data["high"].rolling(params.ma_window, min_periods=params.ma_window).max()
            - data["low"].rolling(params.ma_window, min_periods=params.ma_window).min()
        )
        out["compression"] = out["fast_span"] / out["slow_span"].where(out["slow_span"] > 0.0)
        out["price_change"] = close.diff(params.catchup_bars)
        out["ma_change"] = out["ma"].diff(params.catchup_bars)
        out["z_change"] = out["z"].diff(params.catchup_bars)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (close > out["ma200"])
            & (out["compression"] <= params.compression_ratio)
            & out["z"].between(params.z_floor, params.z_ceiling)
            & (out["price_change"] <= 0.0)
            & (out["ma_change"] < out["price_change"])
            & (out["z_change"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShelfParams,
    ) -> SignalFrame:
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Emit a flat bar even if another entry qualifies today.
                    exit_bar = -1
            elif entry:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=13)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    for holding in (3, 4, 5):
        params = ShelfParams(holding_bars=holding)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
        expected = [0] + [int(i % (holding + 1) < holding) for i in range(len(index) - 1)]
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0.0).all()
