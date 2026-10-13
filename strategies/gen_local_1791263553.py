from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 3
    acceleration_bars: int = 4
    atr_bars: int = 14
    acceleration_threshold: float = 0.65
    coherence_threshold: float = 0.65
    dominance_limit: float = 0.60
    trailing_atr: float = 2.0
    holding_bars: int = 4
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791263553"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.atr_bars + 1, params.roc_bars + params.acceleration_bars + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.roc_bars, params.acceleration_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive")
        if not (params.acceleration_threshold > 0 and params.trailing_atr > 0):
            raise ValueError("Acceleration threshold and trailing multiplier must be positive")
        if not (0 < params.coherence_threshold <= 1 and 0 < params.dominance_limit <= 1):
            raise ValueError("Coherence and dominance limits must lie in (0, 1]")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["absolute_acceleration"] = out["acceleration"].abs()
        window = params.acceleration_bars
        out["net_acceleration"] = out["acceleration"].rolling(window, min_periods=window).sum()
        out["gross_acceleration"] = out["absolute_acceleration"].rolling(window, min_periods=window).sum()
        out["peak_acceleration"] = out["absolute_acceleration"].rolling(window, min_periods=window).max()
        gross = out["gross_acceleration"].replace(0.0, np.nan)
        out["coherence"] = out["net_acceleration"] / gross
        out["dominance"] = out["peak_acceleration"] / gross
        out["atr_fraction"] = out["atr"] / close.replace(0.0, np.nan)
        out["normalized_acceleration"] = out["net_acceleration"] / out["atr_fraction"].replace(0.0, np.nan)
        out["strength"] = out["normalized_acceleration"].abs() * out["coherence"].abs()
        out = out.replace([np.inf, -np.inf], np.nan)
        required = ["atr", "roc", "coherence", "dominance", "normalized_acceleration", "strength"]
        out["ready"] = out[required].notna().all(axis=1) & close.gt(0) & out["atr"].gt(0)
        eligible = (
            out["ready"]
            & out["coherence"].abs().ge(params.coherence_threshold)
            & out["dominance"].le(params.dominance_limit)
            & out["normalized_acceleration"].abs().ge(params.acceleration_threshold)
            & (out["roc"] * out["net_acceleration"]).gt(0)
        )
        out["candidate"] = np.where(eligible, np.sign(out["net_acceleration"].fillna(0.0)), 0).astype(int)
        return out.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, float(params.min_size))
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        strengths = indicators["strength"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        entry_size = float(params.min_size)
        for i in range(count):
            close = closes[i]
            atr = atrs[i]
            if not ready[i] or not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                continue
            if position:
                # The prior close's entry decision fills on this bar's open.
                if i == entry_bar:
                    high_water = close
                    low_water = close
                    stop = close - position * params.trailing_atr * atr
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trailing_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the high-close long trail with a low-close short trail.
                    stop = min(stop, low_water + params.trailing_atr * atr)
                    stopped = close >= stop
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    position = 0
                    sizes[i] = entry_size
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i + 1
                high_water = np.nan
                low_water = np.nan
                stop = np.nan
                exposure = np.clip(strengths[i] / (2.0 * params.acceleration_threshold), 0.0, 1.0)
                entry_size = float(params.min_size + (params.max_size - params.min_size) * exposure)
            signals[i] = position
            sizes[i] = entry_size if position else float(params.min_size)
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(float(params.min_size)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": [100., 101., 104., 100., 99., 98., 97., 95., 99., 100.]}, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "ready": True,
        "candidate": [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        "strength": 1.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    weaker = indicators.copy()
    weaker["strength"] = 0.5
    smaller = strategy.generate_signals(data, weaker, None, params).data
    assert smaller["size"].iloc[1] < result["size"].iloc[1]
    flat = data.copy()
    flat["close"] = 100.0
    expiry = strategy.generate_signals(flat, indicators, None, params).data
    assert expiry["signal"].iloc[4] == 1 and expiry["signal"].iloc[5] == 0


if __name__ == "__main__":
    _self_check()
