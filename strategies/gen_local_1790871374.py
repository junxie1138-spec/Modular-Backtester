from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drift_window: int = 15
    turn_window: int = 3
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790871374"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.drift_window + 1, params.turn_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.drift_window, params.turn_window, params.atr_window, params.max_hold) < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("Lookbacks, holding limit, and ATR multiplier must be positive.")
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["return"] = returns
        out["ma200"] = close.rolling(200).mean()
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["displacement"] = returns.rolling(params.drift_window).sum()
        out["majority_drift"] = returns.rolling(params.drift_window).median()
        out["turn"] = returns.rolling(params.turn_window).sum()
        out["previous_turn"] = out["turn"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1) & np.isfinite(close) & (close > 0) & (atr > 0)
        regime = (data["close"] > indicators["ma200"]) & (indicators["majority_drift"] > 0)
        rebound = (indicators["displacement"] < 0) & (indicators["return"] > 0)
        continuation = (indicators["displacement"] > 0) & (indicators["turn"] > 0) & (indicators["previous_turn"] <= 0)
        entry = (regime & (rebound | continuation)).to_numpy(dtype=bool) & ready
        eligible = regime.to_numpy(dtype=bool) & ready
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                held += 1
                if ready[i]:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not eligible[i] or close[i] <= stop or held >= params.max_hold:
                    active = False
            elif entry[i]:
                active = True
                held = 0
                # Entry fills next bar; its close begins the in-trade high-water mark.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 103., 104., 103., 100., 100.]}, index=index)
    features = pd.DataFrame({
        "return": 0.01,
        "ma200": 90.0,
        "atr": [1., 1., 1., 3., 3., 3., 3.],
        "displacement": 0.01,
        "majority_drift": 0.001,
        "turn": 0.01,
        "previous_turn": -0.01,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, features, ctx, params).data
    # Rising ATR cannot loosen the stop; entry and exit both fill one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    flat = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.}, index=pd.date_range("2020-01-01", periods=240))
    indicators = strategy.indicators(flat, params)
    assert indicators.index.equals(flat.index)
    assert strategy.generate_signals(flat, indicators, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(Params(drift_window=250)) == 251


if __name__ == "__main__":
    _self_check()
