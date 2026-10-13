from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    energy_window: int = 12
    correlation_window: int = 24
    coarse_share: float = 0.55
    min_correlation: float = 0.10
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791206440"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.energy_window + 10,
                       params.correlation_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.energy_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError("Rolling windows must be at least two.")
        if not (0 < params.coarse_share < 1 and 0 <= params.min_correlation < 1):
            raise ValueError("Invalid energy or correlation threshold.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError("Exit parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        price = data["close"].where(data["close"] > 0)
        out["log_price"] = np.log(price)
        for span in (2, 4, 8):
            half = span // 2
            mean = out["log_price"].rolling(half, min_periods=half).mean()
            out[f"haar_{span}"] = (mean - mean.shift(half)) * np.sqrt(half / 2.0)
            out[f"energy_{span}"] = out[f"haar_{span}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["coarse_fraction"] = out["energy_8"] / total.where(total > 0)
        out["energy_migration"] = out["coarse_fraction"].diff(3)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = np.log(opening / price.shift(1))
        out["gap_lag"] = out["gap"].shift(1)
        window = params.correlation_window
        out["gap_mean"] = out["gap"].rolling(window, min_periods=window).mean()
        out["gap_correlation"] = out["gap"].rolling(window, min_periods=window).corr(
            out["gap_lag"]
        ).clip(-1.0, 1.0)
        out["gap_forecast"] = out["gap_mean"] + out["gap_correlation"] * (
            out["gap"] - out["gap_mean"]
        )
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        wavelet_agrees = (
            (indicators["coarse_fraction"] > params.coarse_share)
            & (indicators["energy_migration"] > 0)
            & (indicators["haar_8"] > 0)
        )
        gaps_agree = (
            (indicators["gap_correlation"] > params.min_correlation)
            & (indicators["gap_forecast"] > 0)
        )
        ready = np.isfinite(indicators["atr"]) & (indicators["atr"] > 0)
        entries = (wavelet_agrees & gaps_agree & ready).fillna(False).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry = stop = 0.0
        age = 0
        for i in range(len(data)):
            # Yesterday's target fills at today's open; no future open is read.
            if not active and i > 0 and decisions[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1])):
                    continue
                entry = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                active, armed, age = True, False, 0
            if active:
                age += 1
                # Stops are evaluated at the close; liquidation fills next open.
                # Check the existing stop before raising it with today's close.
                if (not np.isfinite(closes[i]) or not np.isfinite(lows[i])
                        or not np.isfinite(atr[i]) or lows[i] <= stop
                        or age >= params.max_hold):
                    active = False
                    continue
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
                decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0,
        "low": [99.0, 99.0, 99.0, 101.0, 99.0, 99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 102.0, 103.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0
    }, index=index)
    ind = pd.DataFrame({
        "coarse_fraction": [0.8] + [0.0] * 7,
        "energy_migration": 0.1, "haar_8": 0.1,
        "gap_correlation": 0.5, "gap_forecast": 0.01, "atr": 1.0
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, Params(max_hold=5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, Params(max_hold=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    result = strategy.generate_signals(flat, strategy.indicators(flat, Params()),
                                       None, Params()).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
