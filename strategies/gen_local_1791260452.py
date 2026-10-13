from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 60
    gap_quantile: float = 0.75
    volume_quantile: float = 0.80
    profit_target: float = 0.008
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791260452"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.lookback < 2 or not (
            0.0 < params.gap_quantile < 1.0
            and 0.0 < params.volume_quantile < 1.0
        ):
            raise ValueError("Invalid lookback or quantile")
        out = pd.DataFrame(index=data.index)
        prices = data[["open", "close"]].replace([np.inf, -np.inf], np.nan)
        prices = prices.where(prices > 0.0)
        volume = data["volume"].replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0.0)
        out["prior_close"] = prices["close"].shift(1)
        out["gap"] = prices["open"] / out["prior_close"] - 1.0
        out["repair_return"] = prices["close"] / out["prior_close"] - 1.0
        out["gap_threshold"] = (
            out["gap"].abs().rolling(params.lookback).quantile(params.gap_quantile).shift(1)
        )
        out["volume_threshold"] = (
            volume.rolling(params.lookback).quantile(params.volume_quantile).shift(1)
        )
        confirmed = (
            (out["gap"].abs() > out["gap_threshold"])
            & (volume > out["volume_threshold"])
        )
        out["entry"] = np.select(
            [
                confirmed & (out["gap"] < 0.0) & (out["repair_return"] > 0.0),
                confirmed & (out["gap"] > 0.0) & (out["repair_return"] < 0.0),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.max_hold not in (1, 2) or not (
            np.isfinite(params.profit_target) and params.profit_target > 0.0
        ):
            raise ValueError("Use a positive profit target and a one- or two-bar hold")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if direction:
                # The pending entry fills at today's open, observed by today's close.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    direction = 0
                    entry_price = np.nan
                else:
                    gain = direction * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        # Close-observed target or expiry schedules a next-open exit.
                        direction = 0
                        entry_price = np.nan
            elif entries[i] in (-1, 1):
                direction = int(entries[i])
                entry_price = np.nan
            raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 98.0,
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": 100.0,
        },
        index=index,
    )
    strategy = GeneratedStrategy()
    params = GeneratedParams(lookback=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    entries = pd.DataFrame({"entry": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, 0, 0]
    assert (result["size"] > 0.0).all()
    # A next-open gap must change the target's entry basis.
    gapped = data.copy()
    gapped.loc[index[1], "open"] = 110.0
    gapped.loc[index[1], "close"] = 101.0
    assert strategy.generate_signals(gapped, entries, ctx, params).data["signal"].iloc[2] == 1
    before = strategy.indicators(data, params)
    changed = data.copy()
    changed.loc[index[-1], ["close", "volume"]] = [120.0, 10000.0]
    after = strategy.indicators(changed, params)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert before["entry"].iloc[:strategy.warmup_bars(params)].eq(0).all()


if __name__ == "__main__":
    _self_check()
