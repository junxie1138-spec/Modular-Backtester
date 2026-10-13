from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumEnvelopeParams:
    roc_bars: int = 3
    envelope_bars: int = 7
    noise_bars: int = 10
    acceleration_multiple: float = 1.25


class GeneratedStrategy(BaseStrategy[MomentumEnvelopeParams]):
    strategy_id = "gen_local-w2_1791312400"

    @classmethod
    def params_type(cls) -> type[MomentumEnvelopeParams]:
        return MomentumEnvelopeParams

    def warmup_bars(self, params: MomentumEnvelopeParams) -> int:
        return int(params.roc_bars + max(params.envelope_bars + 1, params.noise_bars + 2))

    def indicators(self, data: pd.DataFrame, params: MomentumEnvelopeParams) -> pd.DataFrame:
        if min(params.roc_bars, params.envelope_bars, params.noise_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.acceleration_multiple) or params.acceleration_multiple <= 0:
            raise ValueError("Acceleration multiple must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(periods=params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["noise"] = out["acceleration"].abs().rolling(params.noise_bars, min_periods=params.noise_bars).mean().shift(1)
        out["upper"] = out["roc"].rolling(params.envelope_bars, min_periods=params.envelope_bars).max().shift(1)
        out["lower"] = out["roc"].rolling(params.envelope_bars, min_periods=params.envelope_bars).min().shift(1)
        out["threshold"] = params.acceleration_multiple * out["noise"]
        valid = np.isfinite(out).all(axis=1)
        long_entry = valid & (out["roc"] > out["upper"]) & (out["roc"] > 0) & (out["acceleration"] > out["threshold"])
        short_entry = valid & (out["roc"] < out["lower"]) & (out["roc"] < 0) & (out["acceleration"] < -out["threshold"])
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumEnvelopeParams) -> SignalFrame:
        events = indicators["entry"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                position = 0
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = MomentumEnvelopeParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, 0, -1, -1, 0, 1, 1, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    mirrored = indicators.assign(entry=-indicators["entry"])
    reverse = strategy.generate_signals(data, mirrored, None, params).data
    assert reverse["signal"].tolist() == (-result["signal"]).tolist()
    assert (result["size"] > 0).all()
