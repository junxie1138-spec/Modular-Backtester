from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelayParams:
    rank_window: int = 30
    rank_threshold: float = 0.80
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RelayParams]):
    strategy_id = "gen_local_1791266969"

    @classmethod
    def params_type(cls):
        return RelayParams

    def warmup_bars(self, params: RelayParams) -> int:
        return int(max(params.rank_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RelayParams) -> pd.DataFrame:
        if not (
            params.rank_window >= 2 and params.atr_window >= 2
            and 0.0 < params.rank_threshold <= 1.0
            and np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0.0
            and np.isfinite(params.trail_atr) and params.trail_atr > 0.0
            and params.hold_bars in (1, 2)
        ):
            raise ValueError("Invalid strategy parameters")
        prices = data[["open", "high", "low", "close"]].astype(float).replace(
            [np.inf, -np.inf], np.nan
        )
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = prices["close"].shift(1)
        out["gap"] = prices["open"] - out["previous_close"]
        out["day_gain"] = prices["close"] - prices["open"]
        out["true_range"] = pd.concat(
            [
                prices["high"] - prices["low"],
                (prices["high"] - out["previous_close"]).abs(),
                (prices["low"] - out["previous_close"]).abs(),
            ], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["relay"] = (
            (out["day_gain"] - out["gap"].abs())
            / out["true_range"].where(out["true_range"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["relay_rank"] = out["relay"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        out["qualified"] = (
            out["relay_rank"].ge(params.rank_threshold)
            & out["relay"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = (
            out["qualified"] & out["qualified"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelayParams,
    ) -> SignalFrame:
        values = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        warmup = self.warmup_bars(params)
        for i, (opening, high, low, close) in enumerate(values):
            valid = (
                np.isfinite(values[i]).all()
                and min(opening, high, low, close) > 0.0
                and low <= min(opening, close) <= max(opening, close) <= high
                and np.isfinite(atr[i]) and atr[i] > 0.0
            )
            if pending:
                pending = False
                if not valid:
                    continue
                # Reference entry is the open receiving yesterday's shifted signal.
                entry = peak = opening
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                # Previously established stops are tested before today's high.
                if not valid or low <= stop:
                    active = False
                    continue
                peak = max(peak, high)
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # A newly raised stop is checked against the closing price.
                # All exits are close decisions executed at the following open.
                if close <= stop or age >= params.hold_bars:
                    active = False
                    continue
                decisions[i] = 1
            elif valid and i >= warmup - 1 and confirmed[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RelayParams(rank_window=2, atr_window=2)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8,
         "close": 100.1, "volume": 1000.0}, index=index
    )
    features = pd.DataFrame(
        {"atr": 1.0, "confirmed": False}, index=index
    )
    features.loc[index[3], "confirmed"] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    data.loc[index[4], ["high", "close"]] = [102.0, 100.0]
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    computed = strategy.indicators(data, params)
    expected = computed["qualified"] & computed["qualified"].shift(1, fill_value=False)
    assert computed["confirmed"].equals(expected)
    assert result["size"].gt(0.0).all()
