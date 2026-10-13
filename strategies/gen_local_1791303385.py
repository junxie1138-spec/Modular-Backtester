from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    frac_order: float = 0.4
    frac_window: int = 48
    normalize_window: int = 60
    gap_window: int = 20
    impulse_threshold: float = 0.4
    gap_threshold: float = 0.5
    counterflow_threshold: float = -0.25
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791303385"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.frac_window + params.normalize_window + 1,
                       params.gap_window + 2))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_window, params.normalize_window, params.gap_window) < 2:
            raise ValueError("lookbacks must be at least two")
        if not (params.impulse_threshold > 0 and params.gap_threshold > 0
                and -1.0 <= params.counterflow_threshold < 0.0):
            raise ValueError("invalid entry thresholds")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_open"] = np.log(data["open"].where(data["open"] > 0))
        # ponytail: finite fractional filter; increase frac_window for longer memory.
        frac = out["log_close"].copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            frac = frac + weight * out["log_close"].shift(lag)
        out["frac_close"] = frac
        out["frac_mean"] = frac.rolling(params.normalize_window).mean().shift(1)
        out["frac_std"] = frac.rolling(params.normalize_window).std().shift(1)
        out["frac_z"] = ((frac - out["frac_mean"])
                         / out["frac_std"].replace(0.0, np.nan))
        out["overnight"] = out["log_open"] - out["log_close"].shift(1)
        out["intraday"] = out["log_close"] - out["log_open"]
        out["gap_scale"] = out["overnight"].rolling(params.gap_window).std().shift(1)
        out["gap_z"] = out["overnight"] / out["gap_scale"].replace(0.0, np.nan)
        out["counterflow"] = out["overnight"].rolling(params.gap_window).corr(out["intraday"])
        out["close_return"] = out["log_close"].diff()
        out["impulse_cross"] = ((out["frac_z"] > params.impulse_threshold)
                                & (out["frac_z"].shift(1) <= params.impulse_threshold))
        eligible = (out["impulse_cross"]
                    & (out["gap_z"] <= -params.gap_threshold)
                    & (out["close_return"] >= 0.0)
                    & (out["counterflow"] <= params.counterflow_threshold))
        out["entry_strength"] = (out["frac_z"].clip(lower=0.0)
                                 + (-out["gap_z"]).clip(lower=0.0)
                                 + (-out["counterflow"]).clip(lower=0.0))
        out["entry_strength"] = (out["entry_strength"].where(eligible, 0.0)
                                 .replace([np.inf, -np.inf], 0.0).fillna(0.0))
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("entry_strength")] = 0.0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        if params.hold_bars < 1 or not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("invalid holding period or sizes")
        strengths = indicators["entry_strength"].fillna(0.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        for i, strength in enumerate(strengths):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif np.isfinite(strength) and strength > 0.0:
                entry_size = (params.min_size + (params.max_size - params.min_size)
                              * float(np.clip(strength / 3.0, 0.0, 1.0)))
                exit_bar = i + params.hold_bars
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_strength": 0.0}, index=index)
    indicators.iloc[[0, 2, 18, 19, 20, 38], 0] = [0.6, 3.0, 3.0, 1.5, 3.0, 3.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    expected = np.zeros(45, dtype=int)
    expected[1:19] = 1
    expected[20:38] = 1
    expected[39:] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert np.allclose(result["size"].iloc[1:19], 0.39)
    assert np.allclose(result["size"].iloc[20:38], 0.60)
    assert np.allclose(result["size"].iloc[39:], 0.95)
    assert (result["size"] > 0.0).all()
    assert (strategy.indicators(data, params)["entry_strength"] == 0.0).all()


if __name__ == "__main__":
    _self_check()
