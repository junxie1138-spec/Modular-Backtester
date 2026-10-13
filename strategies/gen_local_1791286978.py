from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAuctionParams:
    gap_window: int = 12
    rank_window: int = 126
    entry_percentile: float = 0.85
    hold_bars: int = 18

    def __post_init__(self):
        if self.gap_window < 2 or self.rank_window < 2 or self.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and holding period positive")
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")


class GeneratedStrategy(BaseStrategy[GapAuctionParams]):
    strategy_id = "gen_local_1791286978"

    @classmethod
    def params_type(cls):
        return GapAuctionParams

    def warmup_bars(self, params):
        return int(params.gap_window + params.rank_window)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / out["previous_close"] - 1).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = ((data["close"] - data["open"]) / out["previous_close"]).replace([np.inf, -np.inf], np.nan)
        valid = out[["gap", "intraday"]].notna().all(axis=1)
        out["up_gap"] = out["gap"].clip(lower=0).where(valid)
        out["down_gap"] = (-out["gap"]).clip(lower=0).where(valid)
        out["repair"] = out["intraday"].clip(lower=0).clip(upper=out["down_gap"]).where(valid)
        out["giveback"] = (-out["intraday"]).clip(lower=0).clip(upper=out["up_gap"]).where(valid)
        for column in ("up_gap", "down_gap", "repair", "giveback"):
            out[column + "_sum"] = out[column].rolling(params.gap_window, min_periods=params.gap_window).sum()
        out["repair_fraction"] = out["repair_sum"] / out["down_gap_sum"].replace(0, np.nan)
        out["giveback_fraction"] = out["giveback_sum"] / out["up_gap_sum"].replace(0, np.nan)
        out["asymmetry"] = out["repair_fraction"] - out["giveback_fraction"]
        out["percentile"] = out["asymmetry"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["previous_percentile"] = out["percentile"].shift(1)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = (indicators["percentile"].ge(params.entry_percentile) & indicators["previous_percentile"].lt(params.entry_percentile)).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapAuctionParams()
    index = pd.date_range("2020-01-01", periods=2 * params.hold_bars + 6, freq="B")
    rank = pd.Series(0.1, index=index)
    rank.iloc[[1, params.hold_bars + 1, params.hold_bars + 3]] = 0.95
    features = pd.DataFrame({"percentile": rank, "previous_percentile": rank.shift(1)}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(pd.DataFrame(index=index), features, ctx, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[2:2 + params.hold_bars] = 1
    expected[params.hold_bars + 4:2 * params.hold_bars + 4] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
