from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 8
    atr_window: int = 14
    regime_on: float = 0.15
    regime_off: float = 0.03
    innovation_min: float = 0.02
    breakeven_pct: float = 0.004
    trail_atr: float = 1.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791324485"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        values = (p.regime_on, p.regime_off, p.innovation_min,
                  p.breakeven_pct, p.trail_atr)
        if (p.trend_window < 2 or p.atr_window < 1 or
                p.max_hold not in (1, 2) or not np.isfinite(values).all() or
                not 0 <= p.regime_off < p.regime_on or
                p.innovation_min < 0 or p.breakeven_pct <= 0 or p.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out["log_atr"] = (out["atr"] / data["close"]).where(
            (out["atr"] > 0) & (data["close"] > 0)
        )
        n = p.trend_window
        t = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["price_sum"] = out["log_close"].rolling(n, min_periods=n).sum()
        out["time_price_sum"] = (t * out["log_close"]).rolling(n, min_periods=n).sum()
        out["latent_slope"] = (
            out["time_price_sum"] - (t - (n - 1) / 2.0) * out["price_sum"]
        ) / (n * (n * n - 1) / 12.0)
        out["latent_level"] = out["price_sum"] / n + out["latent_slope"] * (n - 1) / 2.0
        out["drift"] = out["latent_slope"] / out["log_atr"]
        out["slope_innovation"] = out["latent_slope"].diff() / out["log_atr"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        raw = np.zeros(len(data), dtype=int)
        regime = False
        confirmations = 0
        active = False
        pending = False
        armed = False
        entry = stop = entry_atr = np.nan
        age = 0
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        features = indicators[["atr", "drift", "slope_innovation"]].to_numpy(dtype=float)
        for i, ((opening, high, low, close), (atr, drift, innovation)) in enumerate(zip(prices, features)):
            valid = np.isfinite([opening, high, low, close, atr, drift, innovation]).all()
            valid = valid and min(opening, high, low, close, atr) > 0
            if pending:
                # The preceding close scheduled entry at this bar's open.
                pending = False
                if valid:
                    active = True
                    entry = opening
                    stop = entry - p.trail_atr * entry_atr
                    armed = False
                    age = 0
            if not valid:
                active = False
                regime = False
                confirmations = 0
                continue
            if drift >= p.regime_on:
                regime = True
            elif drift <= p.regime_off:
                regime = False
            if active:
                age += 1
                # Test the previously known stop before using today's high.
                exit_now = low <= stop
                if not exit_now:
                    if high >= entry * (1.0 + p.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, high - p.trail_atr * atr)
                        exit_now = close <= stop
                if exit_now or age >= p.max_hold or not regime:
                    active = False
                    confirmations = 0
                else:
                    raw[i] = 1
                continue
            if regime and innovation >= p.innovation_min:
                confirmations += 1
            else:
                confirmations = 0
            if confirmations >= 2:
                raw[i] = 1
                pending = True
                entry_atr = atr
                confirmations = 0
        df = pd.DataFrame({"signal": raw, "size": np.ones(len(data), dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8,
                         "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"atr": 1.0, "drift": 0.3,
                             "slope_innovation": 0.1}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1]
    data.loc[index[2], ["high", "close"]] = [101.0, 99.9]
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[2] == 1 and result["signal"].iloc[3] == 0
    index = pd.date_range("2020-01-01", periods=80)
    close = 100.0 + np.arange(80) * 0.1 + np.sin(np.arange(80) / 3.0)
    data = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:50], params)
    assert np.allclose(full.iloc[:50].to_numpy(), prefix.to_numpy(), equal_nan=True)
    a = strategy.generate_signals(data, full, ctx, params).data
    b = strategy.generate_signals(data.iloc[:50], prefix, ctx, params).data
    assert a.iloc[:50].equals(b)
    assert set(a["signal"].unique()) <= {0, 1} and (a["size"] > 0).all()
