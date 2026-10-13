from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterEndParams:
    volume_window: int = 20
    percentile_window: int = 126
    percentile: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[QuarterEndParams]):
    strategy_id = "gen_local-w2_1791339871"

    @classmethod
    def params_type(cls) -> type[QuarterEndParams]:
        return QuarterEndParams

    def warmup_bars(self, params: QuarterEndParams) -> int:
        return int(params.volume_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: QuarterEndParams) -> pd.DataFrame:
        if params.volume_window < 2 or params.percentile_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.percentile < 1.0:
            raise ValueError("Percentile must be strictly between zero and one")
        result = pd.DataFrame(index=data.index)
        result["previous_close"] = data["close"].shift(1)
        result["session_return"] = (
            data["close"] / data["open"].where(data["open"] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        result["volume_baseline"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        result["relative_volume"] = (
            data["volume"] / result["volume_baseline"].where(result["volume_baseline"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        result["participation_impulse"] = (
            result["session_return"] * result["relative_volume"]
        ).replace([np.inf, -np.inf], np.nan)
        result["volume_threshold"] = result["relative_volume"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.percentile).shift(1)
        result["impulse_threshold"] = result["participation_impulse"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.percentile).shift(1)
        result["quarter_end_window"] = (
            data.index.month.isin([3, 6, 9, 12]) & (data.index.day >= 20)
        )
        result["entry"] = (
            result["quarter_end_window"]
            & (result["session_return"] > 0)
            & (data["close"] > result["previous_close"])
            & (result["relative_volume"] > result["volume_threshold"])
            & (result["participation_impulse"] > result["impulse_threshold"])
        ).fillna(False)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuarterEndParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be three, four, or five")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar, enter in enumerate(entries):
            if bar < exit_bar:
                signals[bar] = 1
            elif bar == exit_bar:
                # Enforce the scheduled exit even if a new entry qualifies.
                signals[bar] = 0
            elif enter:
                signals[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check delayed entry, exact four-bar holds, and mandatory exit despite repeated entries.
    sample = pd.DataFrame(index=pd.date_range("2024-03-20", periods=11, freq="B"))
    features = pd.DataFrame({"entry": True}, index=sample.index)
    output = GeneratedStrategy.generate_signals(
        None, sample, features, None, QuarterEndParams()
    ).data
    assert output["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert (output["size"] > 0).all()
