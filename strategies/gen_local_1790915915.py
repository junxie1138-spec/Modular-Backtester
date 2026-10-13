from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    entropy_window: int = 18
    percentile_window: int = 126
    entry_quantile: float = 0.25
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    breakeven_gain: float = 0.005
    trail_atr: float = 1.5
    holding_days: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1790915915"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(params.entropy_window + params.percentile_window + 4,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if min(params.entropy_window, params.percentile_window,
               params.volatility_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.entry_quantile < 1 and 1 <= params.holding_days <= 2):
            raise ValueError("Invalid percentile or holding horizon.")
        controls = [params.target_volatility, params.breakeven_gain, params.trail_atr]
        if not all(np.isfinite(x) and x > 0 for x in controls):
            raise ValueError("Risk parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_lag1"] = out["return"].shift(1)
        out["return_lag2"] = out["return"].shift(2)
        triples = out[["return_lag2", "return_lag1", "return"]].to_numpy()
        order = np.argsort(triples, axis=1, kind="stable")
        out["pattern"] = pd.Series(
            9 * order[:, 0] + 3 * order[:, 1] + order[:, 2],
            index=data.index, dtype=float
        ).where(np.isfinite(triples).all(axis=1))
        entropy = pd.Series(0.0, index=data.index)
        for pattern in (5, 7, 11, 15, 19, 21):
            probability = (out["pattern"] == pattern).astype(float).where(
                out["pattern"].notna()
            ).rolling(params.entropy_window).mean()
            out[f"probability_{pattern}"] = probability
            entropy -= probability * np.log(probability.where(probability > 0, 1.0))
        out["complexity"] = entropy / np.log(6.0)
        out["threshold"] = out["complexity"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        out["direction"] = out["return"].rolling(3).sum()
        out["annual_volatility"] = out["return"].rolling(
            params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["target_size"] = (
            params.target_volatility / out["annual_volatility"].where(
                out["annual_volatility"] > 0
            )
        ).clip(0.05, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        z = indicators
        ready = np.isfinite(z[["complexity", "threshold", "direction",
                              "atr", "target_size"]]).all(axis=1) & (z["atr"] > 0)
        entries = (
            ready & (z["complexity"] <= z["threshold"])
            & (z["complexity"].shift(1) > z["threshold"].shift(1))
            & (z["direction"] > 0) & (z["return"] > 0)
        ).fillna(False).to_numpy()
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = z["atr"].to_numpy(dtype=float)
        target = z["target_size"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        active = pending = armed = False
        entry = stop = peak = 0.0
        held_size = 0.05
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite([opening, high, low, close]).all() and min(
                opening, high, low, close
            ) > 0
            if pending:
                pending = False
                if valid:
                    active = True
                    entry = opening
                    peak = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                else:
                    continue
            if active:
                age += 1
                # Stops are close decisions filled next open, not intrabar fills.
                exit_now = not valid or low <= stop
                if not exit_now:
                    peak = max(peak, high)
                    armed = armed or high >= entry * (1.0 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                        # A newly raised stop is compared with close, not earlier lows.
                        exit_now = close <= stop
                if exit_now or age >= params.holding_days:
                    active = False
                    continue
                decisions[i] = 1
                sizes[i] = held_size
            elif valid and entries[i]:
                pending = True
                held_size = float(target[i])
                decisions[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    index = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.5,
                         "close": 100.8, "volume": 1000.0}, index=index)
    features = pd.DataFrame({
        "complexity": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        "threshold": 0.2, "direction": 0.01, "return": 0.001,
        "atr": 1.0, "target_size": 0.6
    }, index=index)
    result = strategy.generate_signals(bars, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    bars.loc[index[2], "high"] = 103.0
    bars.loc[index[2], "close"] = 101.0
    result = strategy.generate_signals(bars, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    assert (result["size"] > 0).all() and result.loc[index[2], "size"] == 0.6
