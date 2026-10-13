from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 40
    shape_step: int = 5
    memory: int = 252
    horizon: int = 18
    atr_window: int = 14
    bandwidth: float = 0.9
    entry_edge: float = 0.35
    min_mass: float = 1.0
    stop_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791257658"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.ma_window + 2 * params.shape_step + params.memory + 1,
                       params.atr_window + params.memory + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if (p.ma_window < 2 or p.shape_step < 1 or p.horizon < 1
                or p.atr_window < 1 or p.memory < p.horizon + 2 * p.shape_step + 1
                or not np.isfinite([p.bandwidth, p.entry_edge, p.min_mass, p.stop_atr]).all()
                or p.bandwidth <= 0 or not 0 < p.entry_edge < 1
                or p.min_mass <= 0 or p.stop_atr <= 0):
            raise ValueError("Invalid analog parameters")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(p.ma_window, min_periods=p.ma_window).mean()
        out["std"] = close.rolling(p.ma_window, min_periods=p.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].where(out["std"] > 0)
        out["speed"] = out["z"] - out["z"].shift(p.shape_step)
        out["curve"] = out["speed"] - out["speed"].shift(p.shape_step)
        previous = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out["outcome_scale"] = out["atr"] / close.where(close > 0) * np.sqrt(p.horizon)
        mass = pd.Series(0.0, index=data.index)
        weighted = mass.copy()
        squared = mass.copy()
        # Historical outcomes end before the current shape begins.
        # Stepping by horizon keeps candidate outcome windows disjoint.
        for lag in range(p.horizon + 2 * p.shape_step + 1, p.memory + 1, p.horizon):
            distance = sum((out[key] - out[key].shift(lag)) ** 2
                           for key in ("z", "speed", "curve"))
            denominator = close.shift(lag).where(close.shift(lag) > 0)
            outcome = (close.shift(lag - p.horizon) / denominator - 1.0)
            scale = out["outcome_scale"].shift(lag)
            outcome = outcome / scale.where(scale > 0)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            weight = np.exp(-distance / (2.0 * p.bandwidth ** 2)).where(valid, 0.0)
            outcome = outcome.where(valid, 0.0)
            mass += weight
            weighted += weight * outcome
            squared += weight * outcome ** 2
        out["mass"] = mass
        denominator = mass.where(mass > 0)
        out["expected"] = weighted / denominator
        out["rms"] = np.sqrt(squared / denominator)
        out["edge"] = (out["expected"] / out["rms"].where(out["rms"] > 0)).clip(-1.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        p = params
        n = len(data)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        edges = indicators["edge"].to_numpy(dtype=float)
        masses = indicators["mass"].to_numpy(dtype=float)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.2, dtype=float)
        position = 0
        age = 0
        highest = lowest = trail = short_trail = 0.0
        first = self.warmup_bars(p) - 1
        for i in range(n):
            close, atr, edge, mass = closes[i], atrs[i], edges[i], masses[i]
            price_valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if not price_valid:
                    position = 0
                else:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    trail = max(trail, highest - p.stop_atr * atr)
                    short_trail = min(short_trail, lowest + p.stop_atr * atr)
                    # Required high-water exit applies to both directions.
                    # Shorts also have a mirrored adverse-price trailing exit.
                    if (close <= trail or (position < 0 and close >= short_trail)
                            or age >= p.horizon):
                        position = 0
            elif (i >= first and price_valid and np.isfinite(edge)
                  and np.isfinite(mass) and mass >= p.min_mass
                  and abs(edge) >= p.entry_edge):
                position = 1 if edge > 0 else -1
                age = 0
                highest = lowest = close
                trail = close - p.stop_atr * atr
                short_trail = close + p.stop_atr * atr
            signals[i] = position
            if position:
                strength = abs(edge) if np.isfinite(edge) else 0.0
                sizes[i] = float(np.clip(0.2 + 0.75 * strength, 0.2, 0.95))
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = AnalogParams(ma_window=2, shape_step=1, memory=4, horizon=1, atr_window=1)
    index = pd.date_range("2020-01-01", periods=360, freq="B")
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.0002 * t + 0.025 * np.sin(t / 7.0))
    data = pd.DataFrame({"open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": np.full(len(index), 1000000.0)}, index=index)
    default = AnalogParams()
    full = strategy.indicators(data, default)
    prefix = strategy.indicators(data.iloc[:330], default)
    assert np.allclose(full.iloc[:330].to_numpy(), prefix.to_numpy(), equal_nan=True)
    params.horizon = 18
    start = strategy.warmup_bars(params) - 1
    check = data.iloc[:start + 8].copy()
    check.iloc[start:, check.columns.get_loc("close")] = [100., 102., 101., 104., 103., 100., 99., 100.]
    features = pd.DataFrame({"atr": 1.0, "edge": 0.8, "mass": 2.0}, index=check.index)
    result = strategy.generate_signals(check, features, None, params).data
    assert result["signal"].iloc[start:start + 7].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert result["signal"].isin([-1, 0, 1]).all()
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
    features["edge"] = -0.8
    short = strategy.generate_signals(check, features, None, params).data
    assert short["signal"].iloc[start + 1] == -1
    assert short["signal"].iloc[start + 4] == 0


if __name__ == "__main__":
    _self_check()
