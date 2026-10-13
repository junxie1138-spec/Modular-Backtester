from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    noise_window: int = 20
    gap_multiple: float = 1.75
    repair_min: float = 0.5
    hold_bars: int = 2

    def __post_init__(self):
        if self.noise_window < 2 or not np.isfinite(self.gap_multiple) or self.gap_multiple <= 0:
            raise ValueError("Require noise_window >= 2 and finite gap_multiple > 0")
        if not 0 < self.repair_min < 1 or self.hold_bars not in (1, 2):
            raise ValueError("Require 0 < repair_min < 1 and hold_bars in (1, 2)")


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local-w2_1790979485"

    @classmethod
    def params_type(cls):
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return int(max(200, params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["gap"] = data["open"] / out["prior_close"].replace(0, np.nan) - 1.0
        out["gap_noise"] = out["gap"].abs().rolling(
            params.noise_window, min_periods=params.noise_window
        ).median().shift(1).replace(0, np.nan)
        out["repair"] = (data["close"] - data["open"]) / (
            out["prior_close"] - data["open"]
        ).replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRejectionParams,
    ) -> SignalFrame:
        ind = indicators
        eligible = (
            ind.notna().all(axis=1)
            & (ind["gap"].abs() >= params.gap_multiple * ind["gap_noise"])
            & (ind["repair"] >= params.repair_min)
            & (ind["repair"] < 1.0)
        )
        long_entry = (
            eligible
            & (ind["gap"] < 0)
            & (data["open"] < ind["prior_low"])
            & (data["close"] > ind["prior_low"])
            & (data["close"] < ind["prior_close"])
            & (data["close"] > ind["ma200"])
        )
        short_entry = (
            eligible
            & (ind["gap"] > 0)
            & (data["open"] > ind["prior_high"])
            & (data["close"] < ind["prior_high"])
            & (data["close"] > ind["prior_close"])
            & (data["close"] < ind["ma200"])
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        side = 0
        remaining = 0
        for i, candidate in enumerate(entries):
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    side = 0
                # An expiry always emits zero; entries cannot replace this exit.
            elif candidate != 0:
                side = int(candidate)
                remaining = params.hold_bars
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        close = 150.0 + direction * 0.2 * np.arange(230)
        sample = pd.DataFrame({
            "open": close - direction * 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000000.0,
        }, index=pd.date_range("2020-01-01", periods=230, freq="B"))
        prior = sample["close"].iloc[219]
        sample.iloc[220, sample.columns.get_loc("open")] = prior - direction * 2.0
        sample.iloc[220, sample.columns.get_loc("close")] = prior - direction * 0.4
        sample["high"] = sample[["open", "close"]].max(axis=1) + 0.5
        sample["low"] = sample[["open", "close"]].min(axis=1) - 0.5
        for hold in (1, 2):
            params = GapRejectionParams(hold_bars=hold)
            ind = strategy.indicators(sample, params)
            ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
            result = strategy.generate_signals(sample, ind, ctx, params).data
            assert result["signal"].iloc[220:224].tolist() == (
                [0, direction, 0, 0] if hold == 1 else [0, direction, direction, 0]
            )
            assert (result["size"] > 0).all()
