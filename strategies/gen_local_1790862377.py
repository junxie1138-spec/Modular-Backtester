from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReleaseParams:
    lookback: int = 60
    compression_quantile: float = 0.25
    release_quantile: float = 0.80
    profit_target: float = 0.008
    max_hold_bars: int = 2

    def __post_init__(self):
        if self.lookback < 3 or self.max_hold_bars not in (1, 2):
            raise ValueError("lookback must be >= 3; max_hold_bars must be 1 or 2")
        if not (0 < self.compression_quantile < 1 and 0 < self.release_quantile < 1):
            raise ValueError("quantiles must be between zero and one")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")


class GeneratedStrategy(BaseStrategy[RangeReleaseParams]):
    strategy_id = "gen_local_1790862377"

    @classmethod
    def params_type(cls):
        return RangeReleaseParams

    def warmup_bars(self, params: RangeReleaseParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: RangeReleaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["width"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["normalized_range"] = out["width"] / data["close"].where(data["close"] > 0)
        out["normalized_range"] = out["normalized_range"].replace([np.inf, -np.inf], np.nan)
        out["compression_cutoff"] = out["normalized_range"].rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.compression_quantile).shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["prior_width"] = out["width"].shift(1)
        out["inside_day"] = (
            (out["prior_high"] <= data["high"].shift(2))
            & (out["prior_low"] >= data["low"].shift(2))
        )
        out["up_extension"] = (data["high"] - out["prior_high"]).clip(lower=0)
        out["down_extension"] = (out["prior_low"] - data["low"]).clip(lower=0)
        out["release"] = (
            (out["up_extension"] - out["down_extension"]) / out["prior_width"]
        ).replace([np.inf, -np.inf], np.nan)
        out["release_cutoff"] = out["release"].abs().rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.release_quantile).shift(1)
        out["compressed"] = (
            out["inside_day"]
            & (out["normalized_range"].shift(1) <= out["compression_cutoff"].shift(1))
        )
        long_entry = (
            out["compressed"]
            & (out["release"] > out["release_cutoff"])
            & (data["close"] > out["prior_high"])
        )
        short_entry = (
            out["compressed"]
            & (out["release"] < -out["release_cutoff"])
            & (data["close"] < out["prior_low"])
            & (data["close"] > 0)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeReleaseParams,
    ) -> SignalFrame:
        # A decision is filled next bar; its opening price becomes known on that bar.
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, age, entry_price = 0, 0, np.nan
        for i in range(len(data)):
            if side:
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1) if valid else np.nan
                # Targets are observed at close; flatten at the following open.
                if not valid or gain >= params.profit_target or age >= params.max_hold_bars:
                    side, age, entry_price = 0, 0, np.nan
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                age = 0
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeReleaseParams(lookback=3)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    signal = strategy.generate_signals(data, entries, ctx, params).data["signal"]
    assert signal.tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    data.loc[index[4], "close"] = 99.0
    signal = strategy.generate_signals(data, entries, ctx, params).data["signal"]
    assert signal.tolist() == [0, 1, 0, 0, -1, 0, 0, 0]
    params.max_hold_bars = 1
    data["close"] = 100.0
    signal = strategy.generate_signals(data, entries, ctx, params).data["signal"]
    assert signal.tolist() == [0, 1, 0, 0, -1, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived["entry"].eq(0).all()
    assert strategy.warmup_bars(params) == 5
    assert strategy.generate_signals(data.iloc[:0], derived.iloc[:0], ctx, params).data.empty
    data["high"] = [102, 103, 102, 102, 100.5, 103, 104, 104]
    data["low"] = [98, 97, 98, 98, 99.5, 100, 101, 102]
    data["close"] = [100, 100, 100, 100, 100, 102.8, 103, 103]
    derived = strategy.indicators(data, params)
    assert derived["entry"].iloc[5] == 1
    pd.testing.assert_frame_equal(derived.iloc[:6], strategy.indicators(data.iloc[:6], params))
    data.loc[index[5], ["high", "low", "close"]] = [100, 97, 97.2]
    assert strategy.indicators(data, params)["entry"].iloc[5] == -1


if __name__ == "__main__":
    _self_check()
