from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 6
    z_entry: float = 0.8
    compression_ratio: float = 0.65
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790932226"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 3 <= params.ma_window <= 8:
            raise ValueError("ma_window must be between 3 and 8")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.z_entry) or params.z_entry <= 0:
            raise ValueError("z_entry must be finite and positive")
        if not np.isfinite(params.compression_ratio) or not 0 < params.compression_ratio < 1:
            raise ValueError("compression_ratio must be between 0 and 1")
        return 9

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["ma"] = data["close"].rolling(params.ma_window).mean()
        out["std"] = data["close"].rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = (data["close"] - out["ma"]) / out["std"]
        out["ma_drift"] = (out["ma"] - out["ma"].shift(1)).abs() / out["std"]
        out["short_span"] = data["high"].rolling(3).max() - data["low"].rolling(3).min()
        out["prior_span"] = (data["high"].rolling(8).max() - data["low"].rolling(8).min()).shift(1)
        out["compression"] = out["short_span"] / out["prior_span"].replace(0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["z"].shift(1) <= -params.z_entry)
            & (out["z"] > -params.z_entry)
            & (out["z"] < 0)
            & (out["compression"] <= params.compression_ratio)
            & (out["ma_drift"] <= 0.3)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    entries = pd.DataFrame({"entry": True}, index=index)
    for hold, expected in ((1, [0, 1, 0, 1, 0, 1, 0, 1]),
                           (2, [0, 1, 1, 0, 1, 1, 0, 1])):
        result = strategy.generate_signals(data, entries, ctx, Params(hold_bars=hold))
        assert result.data["signal"].tolist() == expected
    assert not strategy.indicators(data, Params())["entry"].any()
