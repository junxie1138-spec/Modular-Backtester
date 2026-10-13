from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    return_threshold: float = 1.0
    absorption_threshold: float = 0.25
    holding_bars: int = 8

    def __post_init__(self):
        if self.lookback < 2 or not 5 <= self.holding_bars <= 10:
            raise ValueError("lookback must be >= 2 and holding_bars between 5 and 10")
        if not np.isfinite(self.return_threshold) or self.return_threshold <= 0:
            raise ValueError("return_threshold must be finite and positive")
        if not 0 < self.absorption_threshold <= 1:
            raise ValueError("absorption_threshold must be in (0, 1]")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791282370"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["return"] = data["close"] / out["previous_close"] - 1.0
        out["gap"] = data["open"] / out["previous_close"] - 1.0
        out["intraday"] = (data["close"] - data["open"]) / out["previous_close"]
        # Positive absorption means down gaps were bought; negative means up gaps were sold.
        out["absorbed_gap"] = -np.sign(out["gap"]) * np.minimum(
            out["gap"].abs(),
            (-np.sign(out["gap"]) * out["intraday"]).clip(lower=0.0),
        )
        window = params.lookback
        out["return_mean"] = out["return"].rolling(window).mean()
        out["return_std"] = out["return"].rolling(window).std()
        out["return_score"] = (
            np.sqrt(window) * out["return_mean"]
            / out["return_std"].replace(0.0, np.nan)
        )
        out["gap_magnitude_sum"] = out["gap"].abs().rolling(window).sum()
        out["absorption_sum"] = out["absorbed_gap"].rolling(window).sum()
        out["absorption_score"] = (
            out["absorption_sum"] / out["gap_magnitude_sum"].replace(0.0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        returns = indicators["return_score"]
        absorption = indicators["absorption_score"]
        long_entry = (returns <= -params.return_threshold) & (
            absorption >= params.absorption_threshold
        )
        short_entry = (returns >= params.return_threshold) & (
            absorption <= -params.absorption_threshold
        )
        entries = np.select(
            [long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position and bar == exit_bar:
                position = 0
                # Emit an explicit flat bar; never re-enter on the expiry decision.
            elif position == 0 and entry:
                position = int(entry)
                exit_bar = bar + params.holding_bars
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=45)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    scores = pd.DataFrame({"return_score": np.nan, "absorption_score": np.nan}, index=index)
    scores.iloc[0] = [-2.0, 0.5]
    scores.iloc[[1, 8, 9]] = [2.0, -0.5]
    actual = strategy.generate_signals(data, scores, ctx, params).data
    assert actual["signal"].tolist() == [0] + [1] * 8 + [0] + [-1] * 8 + [0] * 27
    assert (actual["size"] > 0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params)
    assert (flat.data["signal"] == 0).all()
    assert strategy.warmup_bars(params) == 21
