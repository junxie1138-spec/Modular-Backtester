from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    lookback: int = 20
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791229983"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return 2 * max(5, int(params.lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        n = max(5, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        midpoint = (data["high"] + data["low"]) / 2.0
        out["log_midpoint"] = np.log(midpoint.where(midpoint > 0))
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["time"] = time
        out["midpoint_sum"] = out["log_midpoint"].rolling(n, min_periods=n).sum()
        out["weighted_sum"] = (time * out["log_midpoint"]).rolling(n, min_periods=n).sum()
        out["window_center"] = time - (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        out["slope"] = (
            out["weighted_sum"] - out["window_center"] * out["midpoint_sum"]
        ) / denominator
        out["innovation"] = out["slope"].diff()
        out["innovation_scale"] = (
            out["innovation"].rolling(n, min_periods=n).std(ddof=0).shift(1)
        )
        out["innovation_z"] = out["innovation"] / out["innovation_scale"].where(
            out["innovation_scale"] > 0
        )
        out["channel_high"] = data["high"].rolling(n, min_periods=n).max()
        out["channel_low"] = data["low"].rolling(n, min_periods=n).min()
        out["mean_midpoint"] = midpoint.rolling(n, min_periods=n).mean()
        out["channel_width"] = (
            out["channel_high"] - out["channel_low"]
        ) / out["mean_midpoint"].where(out["mean_midpoint"] > 0)
        out["reference_width"] = out["channel_width"].shift(1).rolling(
            n, min_periods=n
        ).mean()
        out["compression"] = out["channel_width"] / out["reference_width"].where(
            out["reference_width"] > 0
        )
        ready = np.isfinite(out[["slope", "innovation_z", "compression"]]).all(axis=1)
        compressed = ready & (out["compression"] < 0.80)
        long_entry = compressed & (out["slope"] > 0) & (out["innovation_z"] > 1.5)
        short_entry = compressed & (out["slope"] < 0) & (out["innovation_z"] < -1.5)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        hold = max(1, int(params.hold_bars))
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar - entry_bar >= hold:
                    position = 0
            elif entry in (-1, 1):
                position = int(entry)
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    data = pd.DataFrame(index=index)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        indicators = pd.DataFrame({"entry": direction}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        expected = np.array([
            0 if bar == 0 or bar % (params.hold_bars + 1) == 0 else direction
            for bar in range(len(index))
        ])
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert (result["size"] > 0).all()
