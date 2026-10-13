from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    min_forecast: float = 0.006
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = "gen_local_1790879806"

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    def warmup_bars(self, params: StreakAnalogParams) -> int:
        return 524

    def indicators(self, data: pd.DataFrame, params: StreakAnalogParams) -> pd.DataFrame:
        if not np.isfinite(params.min_forecast) or params.min_forecast <= 0:
            raise ValueError("min_forecast must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff()
        up, down = change.gt(0), change.lt(0)
        up_run = up.groupby((~up).cumsum()).cumsum()
        down_run = down.groupby((~down).cumsum()).cumsum()
        # Capped run lengths depend on at most six consecutive changes.
        out["streak"] = (up_run - down_run).clip(-6, 6).where(change.notna())
        for lag in (0, 3, 6, 9):
            out[f"shape_{lag}"] = out["streak"].shift(lag)
        previous = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        # At row t, this is an already observed open-to-open 18-session return.
        out["matured_return"] = (
            data["open"] / data["open"].shift(18).replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        shapes = out[["shape_0", "shape_3", "shape_6", "shape_9"]].to_numpy(dtype=float)
        outcomes = out["matured_return"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        win_rate = np.full(len(data), np.nan)
        counts = np.zeros(len(data), dtype=int)
        # ponytail: bounded 504-bar scan; index discrete signatures if history expands.
        for i in range(self.warmup_bars(params), len(data)):
            if not np.isfinite(shapes[i]).all() or not -5 <= shapes[i, 0] <= -2:
                continue
            candidates = np.arange(i - 19, max(9, i - 504) - 1, -1)
            distance = np.abs(shapes[candidates] - shapes[i]).sum(axis=1)
            valid = (
                np.isfinite(distance)
                & np.isfinite(outcomes[candidates + 19])
                & (shapes[candidates, 0] == shapes[i, 0])
                & (distance <= 4)
            )
            candidates, distance = candidates[valid], distance[valid]
            selected = []
            for rank in np.argsort(distance, kind="stable"):
                j = int(candidates[rank])
                # Prevent overlapping realized holding windows from multiplying votes.
                if all(abs(j - other) >= 18 for other in selected):
                    selected.append(j)
                    if len(selected) == 12:
                        break
            counts[i] = len(selected)
            if len(selected) >= 8:
                realized = outcomes[np.asarray(selected) + 19]
                forecast[i] = float(np.median(realized))
                win_rate[i] = float(np.mean(realized > 0))
        out["forecast"] = forecast
        out["win_rate"] = win_rate
        out["analog_count"] = counts
        out["entry"] = (
            out["forecast"].ge(params.min_forecast)
            & out["win_rate"].ge(2.0 / 3.0)
            & out["analog_count"].ge(8)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakAnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry = np.nan
        stop = peak = 0.0
        armed = False
        entry_bar = -1
        for i in range(self.warmup_bars(params), len(data)):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or np.min(prices[i]) <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                entry = np.nan
                continue
            if active:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry, entry_bar, peak = opening, i, opening
                    stop = max(0.0, entry - params.trail_atr * atr[i - 1])
                    armed = False
                # Stops are close-observed exit decisions, filled at the next open.
                # Check the previously established level before raising it.
                exit_now = low <= stop
                peak = max(peak, high)
                if peak >= entry * 1.02:
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    exit_now = exit_now or close <= stop
                if exit_now or i - entry_bar + 1 >= 18:
                    active = False
                    entry = np.nan
                    continue
                signal[i] = 1
            elif entries[i]:
                active = True
                entry = np.nan
                signal[i] = 1
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), StreakAnalogParams()
    index = pd.date_range("2020-01-01", periods=550, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5,
         "close": 100.25, "volume": 1000000.0}, index=index
    )
    ind = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    ind.iloc[524, ind.columns.get_loc("entry")] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=524)
    timed = strategy.generate_signals(data, ind, ctx, params).data
    assert timed["signal"].iloc[:525].eq(0).all()
    assert timed["signal"].iloc[525:543].eq(1).all()
    assert timed["signal"].iloc[543:].eq(0).all()
    data.iloc[525, data.columns.get_indexer(["high", "close"])] = [103.0, 102.6]
    data.iloc[526, data.columns.get_indexer(["high", "low", "close"])] = [104.0, 101.0, 103.4]
    data.iloc[527, data.columns.get_indexer(["high", "low", "close"])] = [103.0, 101.0, 102.0]
    trailed = strategy.generate_signals(data, ind, ctx, params).data
    assert trailed["signal"].iloc[525:528].eq(1).all()
    assert trailed["signal"].iloc[528:].eq(0).all()
    assert trailed["size"].gt(0).all()
    cycle = np.resize(np.array([-0.003, -0.003, -0.003, 0.004, 0.004, 0.004, 0.004]), 800)
    closes = 100.0 * np.cumprod(1.0 + cycle)
    opens = np.r_[100.0, closes[:-1]]
    history = pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes) * 1.005,
         "low": np.minimum(opens, closes) * 0.995,
         "close": closes, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=800, freq="B"),
    )
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:700], params)
    assert full["entry"].any() and full["analog_count"].max() == 12
    pd.testing.assert_frame_equal(full.iloc[:700], prefix)
    pd.testing.assert_frame_equal(
        trailed.iloc[:540],
        strategy.generate_signals(data.iloc[:540], ind.iloc[:540], ctx, params).data,
    )


if __name__ == "__main__":
    _self_check()
