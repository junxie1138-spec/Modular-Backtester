from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    ma_window: int = 60
    drift_window: int = 10
    atr_window: int = 14
    entry_z: float = 1.8
    max_plasticity: float = 0.3
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.ma_window, self.drift_window, self.atr_window, self.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (self.entry_z > 0 and self.stop_atr > 0 and 0 < self.max_plasticity < 1):
            raise ValueError("Invalid entry, stop, or plasticity threshold.")


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1790899843"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return max(params.ma_window + params.drift_window, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"] / out["std"].where(out["std"] > 0)
        out["anchor_drift"] = out["ma"] - out["ma"].shift(params.drift_window)
        out["plasticity"] = out["anchor_drift"].abs() / out["distance"].abs().where(out["distance"].abs() > 0)
        previous_close = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["strength"] = ((out["z"].abs() / (2 * params.entry_z)) * (1 - out["plasticity"]).clip(0, 1)).clip(0.2, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        plasticity = indicators["plasticity"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        position = 0
        pending = False
        stop = np.nan
        entry_atr = 0.0
        held = 0
        exposure = 1.0

        for i in range(n):
            if pending:
                # The order from yesterday first fills at today's open.
                pending = False
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    position = 0
                    continue
                stop = opening[i] - position * params.stop_atr * entry_atr
                held = 0

            if position:
                held += 1
                stopped = np.isfinite(close[i]) and position * (close[i] - stop) < 0
                if stopped or held >= params.hold_bars:
                    position = 0
                else:
                    signals[i] = position
                    sizes[i] = exposure
                continue

            valid = (np.isfinite(close[i]) and close[i] > 0
                     and np.isfinite(z[i]) and np.isfinite(plasticity[i])
                     and np.isfinite(atr[i]) and atr[i] > 0
                     and np.isfinite(strength[i]))
            if valid and abs(z[i]) >= params.entry_z and plasticity[i] <= params.max_plasticity:
                position = 1 if z[i] < 0 else -1
                entry_atr = atr[i]
                exposure = float(np.clip(strength[i], 0.2, 0.95))
                pending = True
                signals[i] = position
                sizes[i] = exposure

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ElasticParams(stop_atr=2.0, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=5)
    for side in (1, -1):
        data = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0 - side * 5.0, 100.0, 100.0],
            "high": 106.0, "low": 94.0, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({
            "z": [-side * 2.0, 0.0, 0.0, 0.0, 0.0],
            "plasticity": 0.1, "atr": [2.0, 100.0, 100.0, 100.0, 100.0],
            "strength": 0.4,
        }, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0]
        assert result["size"].iloc[1] == 0.4
        # Entry gaps must move the frozen stop to the actual opening fill.
        data.loc[index[1], "open"] = 100.0 + side * 10.0
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0]
        data["close"] = 100.0
        data["open"] = 100.0
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    computed = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, computed, None, params).data
    assert (result["signal"] == 0).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
