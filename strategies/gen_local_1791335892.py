from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMemoryParams:
    memory_window: int = 24
    min_autocorr: float = 0.15
    min_close_location: float = 0.65
    atr_period: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[RangeMemoryParams]):
    strategy_id = "gen_local_1791335892"

    @classmethod
    def params_type(cls):
        return RangeMemoryParams

    def warmup_bars(self, params: RangeMemoryParams) -> int:
        return int(max(params.memory_window + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: RangeMemoryParams) -> pd.DataFrame:
        if params.memory_window < 3 or params.atr_period < 1:
            raise ValueError("Lookback lengths must be positive; memory_window must be at least three.")
        if not (-1.0 <= params.min_autocorr <= 1.0 and 0.0 <= params.min_close_location <= 1.0):
            raise ValueError("Invalid correlation or close-location threshold.")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0.0 and np.isfinite(params.trail_atr) and params.trail_atr > 0.0 and 3 <= params.max_hold <= 5):
            raise ValueError("Invalid stop parameters or holding period.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0.0)
        out["relative_range"] = out["range"] / data["close"].where(data["close"] > 0.0)
        out["range_change"] = np.log(out["relative_range"]).diff()
        out["lagged_change"] = out["range_change"].shift(1)
        out["autocorr"] = out["range_change"].rolling(params.memory_window, min_periods=params.memory_window).corr(out["lagged_change"]).clip(-1.0, 1.0)
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["candidate"] = (
            (out["range_change"] < 0.0)
            & (out["autocorr"] >= params.min_autocorr)
            & (out["close_location"] >= params.min_close_location)
            & (data["close"] > data["open"])
            & (out["atr"] > 0.0)
        ).fillna(False)
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeMemoryParams) -> SignalFrame:
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            open_price, high, low, close = prices[i]
            valid = bool(np.isfinite(prices[i]).all() and (prices[i] > 0.0).all() and np.isfinite(atr[i]) and atr[i] > 0.0)
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    # Entry reference is the next session's open.
                    entry = float(open_price)
                    stop = entry - params.trail_atr * atr[i - 1]
                    peak = entry
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                if not valid or low <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    # Check the old stop first; today's ratchet applies to subsequent bars.
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if close <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif valid and confirmed[i] and (i == 0 or not confirmed[i - 1]):
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeMemoryParams()
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[3], ["high", "low", "close"]] = [102.0, 99.5, 101.0]
    data.loc[index[4], ["high", "low", "close"]] = [101.8, 100.6, 101.2]
    data.loc[index[5], ["high", "low", "close"]] = [101.5, 100.4, 101.0]
    derived = pd.DataFrame({"confirmed": [False, False, True, False, False, False, False, False, False], "atr": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    flat = strategy.indicators(data, params)
    assert not flat["confirmed"].any()
    prefix = strategy.generate_signals(data.iloc[:5], derived.iloc[:5], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:5])


if __name__ == "__main__":
    _self_check()
