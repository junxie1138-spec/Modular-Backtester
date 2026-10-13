from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 9
    regime_window: int = 24
    atr_window: int = 14
    innovation_threshold: float = 0.25
    correlation_threshold: float = 0.15
    stop_atr: float = 1.8
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791275275"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slope_window + params.regime_window + 4,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.slope_window < 3 or params.regime_window < 3
                or params.atr_window < 2 or not 3 <= params.holding_bars <= 5
                or not 0 < params.correlation_threshold < 1
                or not np.isfinite(params.innovation_threshold)
                or params.innovation_threshold <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        weights = np.arange(params.slope_window, dtype=float)
        weights -= weights.mean()
        if len(close):
            slope = np.convolve(close.to_numpy(), weights[::-1],
                                mode="full")[:len(close)] / np.dot(weights, weights)
            slope[:params.slope_window - 1] = np.nan
        else:
            slope = np.empty(0, dtype=float)
        out["slope"] = slope
        out["innovation"] = close.diff() - out["slope"].shift(1)
        out["innovation_lag"] = out["innovation"].shift(1)
        out["correlation"] = out["innovation"].rolling(
            params.regime_window, min_periods=params.regime_window
        ).corr(out["innovation_lag"]).shift(2).clip(-1.0, 1.0)
        out["surprise"] = out["innovation"] / out["atr"]
        out["regime"] = np.select([
            out["correlation"] > params.correlation_threshold,
            out["correlation"] < -params.correlation_threshold,
        ], [1, -1], default=0).astype(int)
        out["candidate"] = (
            np.sign(out["surprise"].fillna(0.0)) * out["regime"]
        ).where(out["surprise"].abs() >= params.innovation_threshold, 0).astype(int)
        out["confirmed"] = (
            out["candidate"].ne(0)
            & out["candidate"].eq(out["candidate"].shift(1))
            & out["regime"].eq(out["regime"].shift(1))
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        confirmed = indicators["confirmed"].to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if position:
                # Entry fills this bar's open after yesterday's decision.
                if not valid:
                    position = 0
                else:
                    if position == 1:
                        extreme = close if np.isnan(extreme) else max(extreme, close)
                        proposed = extreme - params.stop_atr * atr
                        stop = proposed if np.isnan(stop) else max(stop, proposed)
                        hit = close <= stop
                    else:
                        # Mirror the close high-water trail for short positions.
                        extreme = close if np.isnan(extreme) else min(extreme, close)
                        proposed = extreme + params.stop_atr * atr
                        stop = proposed if np.isnan(stop) else min(stop, proposed)
                        hit = close >= stop
                    if hit or i - entry_bar >= params.holding_bars:
                        position = 0
                # Never re-enter on the same close as an exit.
            elif valid and confirmed[i]:
                position = int(candidates[i])
                entry_bar = i
                extreme = np.nan
                stop = np.nan
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=10)
    closes = [100, 100, 103, 106, 104, 102, 100, 99, 101, 101]
    data = pd.DataFrame({
        "open": closes, "high": np.array(closes) + 1,
        "low": np.array(closes) - 1, "close": closes, "volume": 1000,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 1, 4, 1, 1, 1, 1, 1],
        "candidate": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "confirmed": [False, True, True, True, True,
                      False, True, True, True, True],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", 0)
    params = Params(stop_atr=1.5)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, 0]
    flat = data.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    indicators["atr"] = 1.0
    indicators["candidate"] = 1
    indicators["confirmed"] = True
    indicators.iloc[0, indicators.columns.get_loc("confirmed")] = False
    expiry = strategy.generate_signals(flat, indicators, ctx, params).data
    assert expiry["signal"].iloc[2:6].eq(1).all()
    assert expiry["signal"].iloc[6] == 0
    derived = strategy.indicators(flat, params)
    assert derived.index.equals(flat.index)
    assert derived["candidate"].eq(0).all()
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
