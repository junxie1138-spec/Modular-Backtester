from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    gap_window: int = 5
    entry_gap_bps: float = 4.0


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1790866191"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        if not 3 <= params.gap_window <= 9:
            raise ValueError("gap_window must be between 3 and 9")
        if not np.isfinite(params.entry_gap_bps) or params.entry_gap_bps <= 0:
            raise ValueError("entry_gap_bps must be finite and positive")
        return int(params.gap_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        self.warmup_bars(params)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        gaps = (10000.0 * (data["open"] / previous_close - 1.0)).replace(
            [np.inf, -np.inf], np.nan
        )
        window = gaps.rolling(params.gap_window, min_periods=params.gap_window)
        trimmed_gap = (
            window.sum() - window.max() - window.min()
        ) / (params.gap_window - 2)
        return pd.DataFrame(
            {"gap_bps": gaps, "trimmed_gap_bps": trimmed_gap}, index=data.index
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParams,
    ) -> SignalFrame:
        # The position exits only when this same entry condition becomes false.
        entry = indicators["trimmed_gap_bps"].le(-params.entry_gap_bps).fillna(False)
        df = pd.DataFrame({"signal": entry.astype(int), "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    gaps = np.array([0.0, -0.001, -0.001, -0.05, 0.01, -0.001,
                     -0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    opening = 100.0 * (1.0 + gaps)
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, 100.0),
         "low": np.minimum(opening, 100.0), "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(gaps)),
    )
    strategy, params = GeneratedStrategy(), GapParams()
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    assert strategy.warmup_bars(params) == 6
    assert np.isclose(derived["trimmed_gap_bps"].iloc[5], -10.0)
    assert signals["signal"].iloc[:6].eq(0).all()
    assert signals["signal"].iloc[6] == 1 and signals["signal"].iloc[9] == 0
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
    prefix = data.iloc[:9]
    prefix_signals = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params
    ).data
    pd.testing.assert_frame_equal(signals.iloc[:9], prefix_signals)


if __name__ == "__main__":
    _self_check()
