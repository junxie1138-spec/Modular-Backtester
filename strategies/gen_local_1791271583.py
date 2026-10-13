from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSpacingParams:
    rank_window: int = 63
    tail_percentile: float = 0.15
    spacing_ratio: float = 1.5
    max_spacing: int = 15
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapSpacingParams]):
    strategy_id = "gen_local_1791271583"

    @classmethod
    def params_type(cls) -> type[GapSpacingParams]:
        return GapSpacingParams

    def warmup_bars(self, params: GapSpacingParams) -> int:
        if params.rank_window < 2 or params.max_spacing < 2:
            raise ValueError("Lookback and spacing limits must be at least two.")
        if not 0.0 < params.tail_percentile < 0.5:
            raise ValueError("tail_percentile must lie between zero and 0.5.")
        if not np.isfinite(params.spacing_ratio) or params.spacing_ratio <= 1.0:
            raise ValueError("spacing_ratio must be finite and greater than one.")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between five and ten sessions.")
        return int(max(200, params.rank_window + 1 + 2 * params.max_spacing))

    def indicators(self, data: pd.DataFrame, params: GapSpacingParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / out["previous_close"] - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_rank"] = out["gap"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["shock"] = ((out["gap"] < 0.0) & (out["gap_rank"] <= params.tail_percentile)).fillna(False)
        out["shock_bar"] = out["bar_number"].where(out["shock"])
        out["previous_shock_bar"] = out["shock_bar"].ffill().shift(1)
        out["spacing"] = (out["bar_number"] - out["previous_shock_bar"]).where(out["shock"])
        out["previous_spacing"] = out["spacing"].ffill().shift(1)
        out["entry"] = (
            out["shock"]
            & out["spacing"].between(1, params.max_spacing)
            & out["previous_spacing"].between(1, params.max_spacing)
            & (out["spacing"] >= params.spacing_ratio * out["previous_spacing"])
            & (data["close"] > out["ma200"])
            & (out["bar_number"] >= warmup - 1)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSpacingParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, trigger in enumerate(entry):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Expiry takes precedence over another entry on this bar.
                    exit_bar = -1
            elif trigger:
                exit_bar = i + params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapSpacingParams()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=24))
    indicators = pd.DataFrame({"entry": True}, index=data.index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] + [1] * 8 + [0] + [1] * 5
    assert (result["size"] > 0.0).all()
