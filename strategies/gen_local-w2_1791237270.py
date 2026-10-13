from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapIgnitionParams:
    reference_bars: int = 63
    quiet_bars: int = 3
    gap_percentile: float = 0.85
    hold_bars: int = 2

    def __post_init__(self):
        if self.reference_bars < 2 or self.quiet_bars < 1:
            raise ValueError("Lookbacks must be positive; reference_bars must be >= 2.")
        if not 0.5 < self.gap_percentile < 1.0:
            raise ValueError("gap_percentile must be between 0.5 and 1.")
        if self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")


class GeneratedStrategy(BaseStrategy[GapIgnitionParams]):
    strategy_id = "gen_local-w2_1791237270"

    @classmethod
    def params_type(cls):
        return GapIgnitionParams

    def warmup_bars(self, params: GapIgnitionParams) -> int:
        return params.reference_bars + params.quiet_bars + 2

    def indicators(self, data: pd.DataFrame, params: GapIgnitionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["gap"] = data["open"].div(out["prior_close"].where(out["prior_close"] > 0)) - 1.0
        out["abs_gap"] = out["gap"].abs()
        out["gap_threshold"] = out["abs_gap"].shift(1).rolling(params.reference_bars).quantile(params.gap_percentile)
        out["quiet_max"] = out["abs_gap"].shift(1).rolling(params.quiet_bars).max()
        # Freeze the quiet-period baseline before that period begins.
        out["quiet_threshold"] = out["abs_gap"].shift(params.quiet_bars + 1).rolling(params.reference_bars).quantile(0.5)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapIgnitionParams,
    ) -> SignalFrame:
        eligible = (
            indicators.notna().all(axis=1)
            & (indicators["gap"] > indicators["gap_threshold"])
            & (indicators["quiet_max"] < indicators["quiet_threshold"])
            & (data["close"] >= data["open"])
            & (data["close"] > indicators["prior_high"])
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(eligible):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                # Force the scheduled exit even if a new setup occurs.
                continue
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"gap": 0.02, "gap_threshold": 0.01, "quiet_max": 0.001,
         "quiet_threshold": 0.005, "prior_high": 100.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    for hold in (1, 2):
        params = GapIgnitionParams(hold_bars=hold)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        actual = strategy.generate_signals(data, indicators, ctx, params).data
        expected = [0] + [int(i % (hold + 1) < hold) for i in range(7)]
        assert actual["signal"].tolist() == expected
        assert actual["size"].gt(0).all()
        missing = indicators * np.nan
        assert strategy.generate_signals(data, missing, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
