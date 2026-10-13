from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_bars: int = 40
    baseline_bars: int = 15
    shelf_bars: int = 3
    min_drawdown: float = 0.015
    max_drawdown: float = 0.08
    compression_limit: float = 0.85
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local_1791332397"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        return int(max(params.peak_bars, params.baseline_bars + params.shelf_bars + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        if min(params.peak_bars, params.baseline_bars, params.shelf_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0 < params.min_drawdown < params.max_drawdown < 1:
            raise ValueError("Drawdown bounds must satisfy 0 < minimum < maximum < 1.")
        if not 0 < params.compression_limit <= 1:
            raise ValueError("Compression limit must be in (0, 1].")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("Holding period must be three, four, or five bars.")

        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1)
        out["baseline_range"] = out["true_range"].rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(params.shelf_bars)
        out["peak"] = data["high"].rolling(
            params.peak_bars, min_periods=params.peak_bars
        ).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["shelf_high"] = data["high"].rolling(
            params.shelf_bars, min_periods=params.shelf_bars
        ).max()
        out["shelf_low"] = data["low"].rolling(
            params.shelf_bars, min_periods=params.shelf_bars
        ).min()
        out["shelf_width"] = out["shelf_high"] - out["shelf_low"]
        out["compression"] = out["shelf_width"] / out["baseline_range"].where(
            out["baseline_range"] > 0
        )
        out["setup"] = (
            out["drawdown"].between(params.min_drawdown, params.max_drawdown)
            & out["compression"].between(0.0, params.compression_limit)
            & np.isfinite(out[["drawdown", "compression"]]).all(axis=1)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShelfParams,
    ) -> SignalFrame:
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("Holding period must be three, four, or five bars.")
        confirmed = indicators["confirmed"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = None
        for i, ready in enumerate(confirmed):
            if entry_bar is not None:
                if i - entry_bar == params.holding_bars:
                    # Leave this bar flat even if another setup is confirmed.
                    entry_bar = None
                else:
                    signal[i] = 1
            elif ready:
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    setup = pd.Series(True, index=index)
    indicators = pd.DataFrame({"confirmed": setup & setup.shift(1, fill_value=False)}, index=index)
    for holding in (3, 4, 5):
        params = ShelfParams(holding_bars=holding)
        result = strategy.generate_signals(data, indicators, None, params).data
        signal = result["signal"].to_numpy()
        assert signal[:2].tolist() == [0, 0]
        assert signal[2:2 + holding].tolist() == [1] * holding
        assert signal[2 + holding] == 0
        assert signal[3 + holding] == 1
        assert set(signal).issubset({0, 1})
        assert result["size"].gt(0).all()
    derived = strategy.indicators(data, ShelfParams())
    assert not derived["confirmed"].any()


if __name__ == "__main__":
    _self_check()
