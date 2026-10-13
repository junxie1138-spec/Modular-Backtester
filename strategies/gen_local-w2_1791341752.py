from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    ma_bars: int = 20
    history_bars: int = 126
    shelf_bars: int = 3
    breakout_percentile: float = 0.85
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local-w2_1791341752"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        if min(params.ma_bars, params.history_bars, params.shelf_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.5 < params.breakout_percentile < 1.0:
            raise ValueError("breakout_percentile must be between 0.5 and 1")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        return int(params.ma_bars + params.history_bars + params.shelf_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["dispersion"] = close.rolling(params.ma_bars).std(ddof=0).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["dispersion"]).replace([np.inf, -np.inf], np.nan)
        out["ceiling"] = out["z"].shift(1).rolling(params.history_bars).quantile(params.breakout_percentile)
        out["median"] = out["z"].shift(1).rolling(params.history_bars).quantile(0.5)
        out["shelf_low"] = out["z"].shift(1).rolling(params.shelf_bars).min()
        out["shelf_high"] = out["z"].shift(1).rolling(params.shelf_bars).max()
        out["shelf_span"] = out["shelf_high"] - out["shelf_low"]
        out["prior_ceiling"] = out["ceiling"].shift(1)
        out["prior_median"] = out["median"].shift(1)
        out["percentile_band"] = out["prior_ceiling"] - out["prior_median"]
        out["entry"] = (
            (out["shelf_low"] > 0.0)
            & (out["shelf_low"] >= out["prior_median"])
            & (out["shelf_high"] <= out["prior_ceiling"])
            & (out["percentile_band"] > 0.0)
            & (out["shelf_span"] <= 0.5 * out["percentile_band"])
            & (out["z"].shift(1) <= out["ceiling"].shift(1))
            & (out["z"] > out["ceiling"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShelfParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if bar < exit_bar:
                signal[bar] = 1
            elif bar == exit_bar:
                continue
            elif entries[bar]:
                signal[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, True, False, False, True, True, False, False, False, True, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, ShelfParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
