from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 5
    slow_window: int = 13
    volatility_window: int = 20
    percentile_window: int = 63
    entry_quantile: float = 0.85
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8

    def __post_init__(self):
        if not (3 <= self.fast_window < self.slow_window):
            raise ValueError("Require 3 <= fast_window < slow_window")
        if min(self.volatility_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError("Rolling windows must be at least two")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one")
        if not (np.isfinite(self.breakeven_pct) and self.breakeven_pct > 0.0
                and np.isfinite(self.trail_atr) and self.trail_atr > 0.0):
            raise ValueError("Stop parameters must be positive and finite")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791326347"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slow_window, params.fast_window + 1,
                       params.volatility_window + 1, params.atr_window + 1)
                   + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        price = ((data["high"] + data["low"] + data["close"]) / 3.0).where(
            lambda x: x > 0.0)
        out["log_price"] = np.log(price)
        out["log_change"] = out["log_price"].diff()
        out["volatility"] = out["log_change"].rolling(
            params.volatility_window).std().replace(0.0, np.nan)
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        # Exact rolling least-squares slopes without rolling Python callbacks.
        for name, window in (("fast_slope", params.fast_window),
                             ("slow_slope", params.slow_window)):
            total = out["log_price"].rolling(window).sum()
            moment = (time * out["log_price"]).rolling(window).sum()
            out[name] = (moment - (time - (window - 1) / 2.0) * total) / (
                window * (window * window - 1) / 12.0)
        out["prior_fast_slope"] = out["fast_slope"].shift(1)
        out["slope_innovation"] = out["fast_slope"].diff()
        out["innovation_score"] = out["slope_innovation"] / out["volatility"]
        out["innovation_threshold"] = out["innovation_score"].abs().shift(1).rolling(
            params.percentile_window).quantile(params.entry_quantile)
        prior_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prior_close).abs(),
             (data["low"] - prior_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        slow = indicators["slow_slope"]
        fast = indicators["fast_slope"]
        prior = indicators["prior_fast_slope"]
        score = indicators["innovation_score"]
        threshold = indicators["innovation_threshold"]
        valid = np.isfinite(indicators[
            ["slow_slope", "fast_slope", "prior_fast_slope",
             "innovation_score", "innovation_threshold", "atr"]
        ]).all(axis=1) & indicators["atr"].gt(0.0)
        long_entry = valid & slow.gt(0.0) & prior.le(0.0) & fast.gt(0.0) & score.gt(threshold)
        short_entry = valid & slow.lt(0.0) & prior.ge(0.0) & fast.lt(0.0) & score.lt(-threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        ohlc = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        pending = False
        age = 0
        entry = peak = stop = 0.0
        armed = False
        for i, (opening, high, low, close) in enumerate(ohlc):
            if not (np.isfinite([opening, high, low, close]).all()
                    and min(opening, high, low, close) > 0.0):
                direction = 0
                pending = False
                continue
            if pending:
                # The previous close's decision fills at this session's open.
                entry = opening
                peak = direction * entry
                stop = peak - params.trail_atr * atr[i - 1]
                armed = False
                age = 0
                pending = False
            if direction:
                age += 1
                favorable = high if direction == 1 else -low
                adverse = low if direction == 1 else -high
                signed_close = direction * close
                hit = adverse <= stop
                peak = max(peak, favorable)
                armed = armed or peak - direction * entry >= entry * params.breakeven_pct
                if armed and np.isfinite(atr[i]) and atr[i] > 0.0:
                    # Signed stops only rise; short price stops therefore only fall.
                    stop = max(stop, direction * entry, peak - params.trail_atr * atr[i])
                # A newly tightened stop is checked at the close, never retroactively.
                if hit or signed_close <= stop or age >= 2:
                    direction = 0
                decisions[i] = direction
            elif entries[i]:
                direction = int(entries[i])
                pending = True
                decisions[i] = direction
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(trail_atr=1.0)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        for closing, expected in ((99.9, [0, direction, 0, 0]),
                                  (100.6, [0, direction, direction, 0])):
            data = pd.DataFrame(
                {"open": [100.0] * 4, "high": [100.1, 100.8, 101.0, 101.0],
                 "low": [99.9, 99.8, 100.1, 100.1],
                 "close": [100.0, closing, 100.5, 100.5], "volume": [1000.0] * 4},
                index=index)
            if direction == -1:
                data[["open", "close"]] = 200.0 - data[["open", "close"]]
                data[["high", "low"]] = 200.0 - data[["low", "high"]].to_numpy()
            features = pd.DataFrame(
                {"slow_slope": float(direction), "fast_slope": float(direction),
                 "prior_fast_slope": float(-direction),
                 "innovation_score": [2.0 * direction, 0.0, 0.0, 0.0],
                 "innovation_threshold": 1.0, "atr": 1.0}, index=index)
            signals = strategy.generate_signals(data, features, None, params).data
            assert signals["signal"].tolist() == expected
            assert signals["size"].gt(0.0).all()
    empty = data.iloc[:0]
    assert strategy.generate_signals(
        empty, strategy.indicators(empty, params), None, params).data.empty


if __name__ == "__main__":
    _self_check()
