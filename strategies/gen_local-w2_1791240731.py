from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 25
    drift_bars: int = 5
    max_centroid_drift: float = 0.35
    entry_z: float = 1.3
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10
    min_size: float = 0.35
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791240731"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ma_window + params.drift_bars, params.ma_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.ma_window < 2 or params.drift_bars < 1 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive; ma_window must exceed one.")
        if not (params.entry_z > 0 and params.trail_atr > 0 and params.max_centroid_drift >= 0 and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid thresholds or sizing bounds.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["ma"]) / out["dispersion"]
        out["z_turn"] = out["z"].diff()
        out["centroid_drift"] = out["ma"].diff(params.drift_bars).abs() / out["dispersion"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]).abs(), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out[["z", "z_turn", "centroid_drift", "atr"]].notna().all(axis=1) & close.notna() & (out["atr"] > 0)
        out["entry"] = valid & (out["z"] <= -params.entry_z) & (out["z_turn"] > 0) & (out["centroid_drift"] <= params.max_centroid_drift)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        count = len(data)
        signal = np.zeros(count, dtype=int)
        size = np.full(count, params.min_size, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        exposure = params.min_size
        for i in range(count):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(z[i])
            if active:
                if valid:
                    # The first in-trade close occurs one bar after the entry decision.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or z[i] >= 0 or i - entry_bar >= params.max_hold:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = exposure
            elif valid and entries[i]:
                conviction = float(np.clip((-z[i] - params.entry_z) / params.entry_z, 0.0, 1.0))
                exposure = params.min_size + (params.max_size - params.min_size) * conviction
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
                size[i] = exposure
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 102.0, 101.0, 99.0, 99.0]}, index=index)
    derived = pd.DataFrame({"atr": [1.0, 1.0, 5.0, 5.0, 5.0], "z": [-2.0] * 5, "entry": [True, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    expected_size = params.min_size + (params.max_size - params.min_size) * (2.0 - params.entry_z) / params.entry_z
    assert np.isclose(result["size"].iloc[1], expected_size)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
