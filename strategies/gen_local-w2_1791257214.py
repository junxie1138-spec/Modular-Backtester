from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_length: int = 12
    atr_length: int = 14
    innovation_threshold: float = 0.03
    position_ceiling: float = 0.55
    stop_atr: float = 1.0
    hold_bars: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791257214"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.slope_length + 2, params.atr_length + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.slope_length < 3 or params.atr_length < 1
                or params.hold_bars not in (1, 2)
                or not 0.0 < params.position_ceiling < 1.0
                or not np.isfinite(params.innovation_threshold)
                or params.innovation_threshold <= 0.0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0
                or not np.isfinite(params.size) or params.size <= 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        length = params.slope_length
        # Rolling OLS slope using a deterministic time index, without rolling.apply.
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["mean_close"] = close.rolling(length, min_periods=length).mean()
        cross = (time * close).rolling(length, min_periods=length).mean()
        mean_time = time.rolling(length, min_periods=length).mean()
        out["slope"] = (cross - mean_time * out["mean_close"]) / ((length ** 2 - 1) / 12.0)
        out["prior_slope"] = out["slope"].shift(1)
        out["expected_slope"] = 2.0 * out["prior_slope"] - out["slope"].shift(2)
        out["trend_endpoint"] = out["mean_close"] + out["slope"] * ((length - 1) / 2.0)
        out["predicted_close"] = out["trend_endpoint"].shift(1) + out["prior_slope"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_length, min_periods=params.atr_length).mean()
        scale = out["atr"].shift(1).where(out["atr"].shift(1) > 0.0)
        out["slope_innovation"] = (out["slope"] - out["expected_slope"]) / scale
        out["prediction_error"] = (close - out["predicted_close"]) / scale
        out["range_low"] = data["low"].rolling(length, min_periods=length).min()
        out["range_high"] = data["high"].rolling(length, min_periods=length).max()
        span = out["range_high"] - out["range_low"]
        out["range_position"] = (close - out["range_low"]) / span.where(span > 0.0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            (close > out["ma200"])
            & (out["prior_slope"] < 0.0)
            & (out["slope_innovation"] > params.innovation_threshold)
            & (out["prediction_error"] > 0.0)
            & out["range_position"].between(0.0, params.position_ceiling)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        frozen_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    if not np.isfinite(opening[i]):
                        active = False
                        continue
                    # Entry is filled at this bar's open; ATR was known one bar earlier.
                    stop = opening[i] - params.stop_atr * frozen_atr
                age = i - entry_bar + 1
                if (not np.isfinite(close[i]) or close[i] < stop
                        or age >= params.hold_bars):
                    active = False
                else:
                    decisions[i] = 1
            elif (i >= self.warmup_bars(params) - 1 and eligible[i]
                    and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0):
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                decisions[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=205)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({"atr": 2.0, "eligible": False}, index=index)
    derived.loc[index[200], "eligible"] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].iloc[200:205].tolist() == [0, 1, 1, 0, 0]
    # A gap changes the actual entry reference; subsequent ATR changes cannot move the stop.
    data.loc[index[201], ["open", "close"]] = [110.0, 107.0]
    derived.loc[index[201], "atr"] = 50.0
    stopped = strategy.generate_signals(data, derived, ctx, params).data
    assert stopped["signal"].iloc[200:205].tolist() == [0, 1, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert not actual["eligible"].iloc[:199].any()
    assert actual.index.equals(data.index)
    assert result["size"].gt(0.0).all()
    assert result["signal"].isin([0, 1]).all()


if __name__ == "__main__":
    _self_check()
