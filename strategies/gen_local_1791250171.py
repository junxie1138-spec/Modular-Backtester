from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 20
    impulse_bars: int = 3
    min_depth: float = 0.02
    depth_fraction: float = 0.35
    pressure_window: int = 5
    pressure_threshold: float = 0.15
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791250171"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.peak_window + params.impulse_bars,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        # Freeze the peak across the impulse: an expiring peak cannot fake recovery.
        out["anchor"] = data["close"].rolling(
            p.peak_window, min_periods=p.peak_window
        ).max().shift(p.impulse_bars)
        anchor = out["anchor"].where(out["anchor"] > 0)
        out["prior_depth"] = 1.0 - data["close"].shift(p.impulse_bars) / anchor
        out["depth"] = 1.0 - data["close"] / anchor
        out["depth_change"] = out["prior_depth"] - out["depth"]
        spread = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["body_pressure"] = ((data["close"] - data["open"]) / spread).clip(
            -1.0, 1.0
        ).fillna(0.0)
        out["signed_volume"] = out["body_pressure"] * data["volume"]
        out["volume_sum"] = data["volume"].rolling(
            p.pressure_window, min_periods=p.pressure_window
        ).sum()
        out["pressure"] = out["signed_volume"].rolling(
            p.pressure_window, min_periods=p.pressure_window
        ).sum() / out["volume_sum"].where(out["volume_sum"] > 0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            p.atr_window, min_periods=p.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        p = params
        prior = indicators["prior_depth"]
        change = indicators["depth_change"]
        pressure = indicators["pressure"]
        eligible = prior.ge(p.min_depth)
        long_entry = eligible & change.ge(prior * p.depth_fraction) & pressure.ge(
            p.pressure_threshold
        )
        short_entry = eligible & change.le(-prior * p.depth_fraction) & pressure.le(
            -p.pressure_threshold
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        valid = np.isfinite(indicators[
            ["prior_depth", "depth", "depth_change", "pressure", "atr"]
        ].to_numpy()).all(axis=1)
        valid &= indicators["atr"].to_numpy() > 0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry, stop, best = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            if side and np.isnan(entry):
                # The previous decision fills at this bar's open.
                entry = opens[i]
                best = side * entry
                stop = best - p.trail_atr * atrs[i - 1]
                armed = False
                age = 0
            if not valid[i] or not np.isfinite(closes[i]) or (
                side and (not np.isfinite(entry) or entry <= 0)
            ):
                side = 0
                entry = np.nan
                continue
            if side:
                age += 1
                favorable = side * closes[i]
                # Close-confirmed exits execute on the next bar, like entries.
                if favorable <= stop or age >= p.max_hold:
                    side = 0
                    entry = np.nan
                    continue
                best = max(best, favorable)
                if best - side * entry >= entry * p.breakeven_pct:
                    armed = True
                if armed:
                    # Profit-coordinate stop only increases; short price stops fall.
                    stop = max(stop, side * entry, best - p.trail_atr * atrs[i])
            elif candidates[i]:
                side = int(candidates[i])
                entry = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, p = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        close = 100.0 + side * np.array([0, 1, 4, 3, 1, 0], dtype=float)
        opening = np.r_[100.0, close[:-1]]
        data = pd.DataFrame({
            "open": opening, "high": np.maximum(opening, close) + 1,
            "low": np.minimum(opening, close) - 1,
            "close": close, "volume": 1000.0,
        }, index=index)
        indicators = pd.DataFrame({
            "prior_depth": 0.04, "depth": 0.02,
            "depth_change": side * 0.02, "pressure": side * 0.3, "atr": 1.0,
        }, index=index)
        ctx = StrategyContext("SPY", "1d", 0)
        result = strategy.generate_signals(data, indicators, ctx, p).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        prefix = strategy.generate_signals(
            data.iloc[:4], indicators.iloc[:4], ctx, p
        ).data
        assert prefix.equals(result.iloc[:4])
        indicators["pressure"] = 0.0
        assert strategy.generate_signals(
            data, indicators, ctx, p
        ).data["signal"].eq(0).all()
        assert result["size"].gt(0).all()
    flat = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 0.0,
    }, index=pd.date_range("2020-01-01", periods=40))
    computed = strategy.indicators(flat, p)
    assert strategy.generate_signals(flat, computed, ctx, p).data["signal"].eq(0).all()
    assert strategy.warmup_bars(p) >= p.peak_window + p.impulse_bars


if __name__ == "__main__":
    _self_check()
