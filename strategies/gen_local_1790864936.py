from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeFollowthroughParams:
    volume_window: int = 20
    regime_window: int = 63
    volume_multiple: float = 1.15
    followthrough_cutoff: float = 0.58
    min_events: int = 8
    hold_bars: int = 2

    def __post_init__(self):
        if self.volume_window < 2 or self.regime_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 1 <= self.min_events <= self.regime_window:
            raise ValueError("min_events must fit the regime window.")
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive.")
        if not 0.5 < self.followthrough_cutoff < 1:
            raise ValueError("followthrough_cutoff must lie between 0.5 and 1.")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")


class GeneratedStrategy(BaseStrategy[VolumeFollowthroughParams]):
    strategy_id = "gen_local_1790864936"

    @classmethod
    def params_type(cls):
        return VolumeFollowthroughParams

    def warmup_bars(self, params: VolumeFollowthroughParams) -> int:
        return int(params.volume_window + params.regime_window + 3)

    def indicators(self, data: pd.DataFrame, params: VolumeFollowthroughParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = volume.rolling(params.volume_window).mean().shift(1)
        out["relative_volume"] = volume / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["previous_return"] = out["return"].shift(1)
        out["eligible_event"] = (
            out["relative_volume"].shift(1).ge(params.volume_multiple)
            & out["previous_return"].notna()
            & out["return"].notna()
            & out["previous_return"].ne(0)
            & out["return"].ne(0)
        )
        out["successful_event"] = (
            out["eligible_event"]
            & (out["previous_return"] * out["return"]).gt(0)
        )
        out["event_count"] = out["eligible_event"].astype(float).rolling(params.regime_window).sum().shift(1)
        out["success_count"] = out["successful_event"].astype(float).rolling(params.regime_window).sum().shift(1)
        out["followthrough_rate"] = out["success_count"] / out["event_count"].replace(0, np.nan)
        out["ready"] = (
            (np.arange(len(data)) >= self.warmup_bars(params) - 1)
            & out["event_count"].ge(params.min_events)
            & out["followthrough_rate"].notna()
        )
        out["continuation_bar"] = (
            out["ready"]
            & out["followthrough_rate"].ge(params.followthrough_cutoff)
            & out["relative_volume"].ge(params.volume_multiple)
            & out["return"].gt(0)
        )
        out["reversal_bar"] = (
            out["ready"]
            & out["followthrough_rate"].le(1.0 - params.followthrough_cutoff)
            & out["relative_volume"].ge(params.volume_multiple)
            & out["return"].lt(0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeFollowthroughParams,
    ) -> SignalFrame:
        continuation = indicators["continuation_bar"].fillna(False).astype(bool)
        reversal = indicators["reversal_bar"].fillna(False).astype(bool)
        confirmed = (
            (continuation & continuation.shift(1, fill_value=False))
            | (reversal & reversal.shift(1, fill_value=False))
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, entry in enumerate(confirmed):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                # Emit a flat bar even if a fresh entry is confirmed.
                continue
            elif entry:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame(
        {"continuation_bar": True, "reversal_bar": False}, index=index
    )
    for holding in (1, 2):
        params = VolumeFollowthroughParams(hold_bars=holding)
        result = strategy.generate_signals(data, evidence, None, params).data
        expected = [0, 0] + [
            int((i - 2) % (holding + 1) < holding) for i in range(2, 12)
        ]
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
