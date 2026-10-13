from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEnvelopeParams:
    range_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.8
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeEnvelopeParams]):
    strategy_id = "gen_local_1791345210"

    @classmethod
    def params_type(cls) -> type[RangeEnvelopeParams]:
        return RangeEnvelopeParams

    def warmup_bars(self, params: RangeEnvelopeParams) -> int:
        return int(max(params.range_window + params.percentile_window, 2 * params.range_window))

    def indicators(self, data: pd.DataFrame, params: RangeEnvelopeParams) -> pd.DataFrame:
        if params.range_window < 2 or params.percentile_window < 2:
            raise ValueError("Window lengths must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one")
        out = pd.DataFrame(index=data.index)
        window = params.range_window
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["range_sum"] = out["daily_range"].rolling(window, min_periods=window).sum()
        out["upper_envelope"] = data["high"].rolling(window, min_periods=window).max()
        out["lower_envelope"] = data["low"].rolling(window, min_periods=window).min()
        out["envelope_width"] = out["upper_envelope"] - out["lower_envelope"]
        out["efficiency"] = (out["envelope_width"] / out["range_sum"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["efficiency_threshold"] = out["efficiency"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        out["prior_upper_envelope"] = out["upper_envelope"].shift(window)
        out["prior_lower_envelope"] = out["lower_envelope"].shift(window)
        out["entry_ready"] = (
            out["efficiency"].gt(out["efficiency_threshold"])
            & out["upper_envelope"].gt(out["prior_upper_envelope"])
            & out["lower_envelope"].gt(out["prior_lower_envelope"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeEnvelopeParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                else:
                    signals[bar] = 1
            elif entries[bar]:
                signals[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_ready": True}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, RangeEnvelopeParams())
    expected = [0] + ([1] * 8 + [0]) * 2 + [1] * 5
    assert result.data["signal"].tolist() == expected
    assert result.data["size"].gt(0.0).all()
