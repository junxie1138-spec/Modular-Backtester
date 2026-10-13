from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    ma_window: int = 20
    compression_window: int = 6
    baseline_window: int = 40
    phase_window: int = 8
    atr_window: int = 14
    entry_z: float = 0.8
    compression_ratio: float = 0.8
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1791319310"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(200, params.ma_window + params.phase_window + 2,
                       params.baseline_window + 1,
                       params.compression_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["mean"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["mean"]) / out["std"].replace(0.0, np.nan)
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["fast_range"] = out["tr"].rolling(params.compression_window, min_periods=params.compression_window).mean()
        out["slow_range"] = out["tr"].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["compression"] = out["fast_range"] / out["slow_range"].replace(0.0, np.nan)
        out["z_previous"] = out["z"].shift(1)
        out["z_lag2"] = out["z"].shift(2)
        out["phase_product"] = out["z"] * out["z_lag2"]
        out["wave"] = out["phase_product"].rolling(params.phase_window, min_periods=params.phase_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        if params.trail_atr <= 0 or params.hold_bars < 1 or params.entry_z <= 0:
            raise ValueError("trail_atr, hold_bars and entry_z must be positive")
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        required = indicators[["ma200", "atr", "z", "z_previous", "compression", "wave"]]
        valid = pd.Series(np.isfinite(required.to_numpy(dtype=float)).all(axis=1), index=data.index)
        valid &= np.isfinite(data["close"]) & (data["close"] > 0) & (indicators["atr"] > 0)
        gate = valid & (indicators["compression"] <= params.compression_ratio) & (indicators["wave"] < 0)
        long_entry = gate & (data["close"] > indicators["ma200"]) & (indicators["z_previous"] <= -params.entry_z) & (indicators["z"] > indicators["z_previous"]) & (indicators["z"] < 0)
        short_entry = gate & (data["close"] < indicators["ma200"]) & (indicators["z_previous"] >= params.entry_z) & (indicators["z"] < indicators["z_previous"]) & (indicators["z"] > 0)
        entries = np.where(long_entry.to_numpy(), 1, np.where(short_entry.to_numpy(), -1, 0))
        valid_values = valid.to_numpy()
        regime = indicators["ma200"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            if position:
                age += 1
                if not valid_values[i]:
                    position = 0
                elif position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    if close[i] <= stop or age >= params.hold_bars or close[i] <= regime[i]:
                        position = 0
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    if close[i] >= stop or age >= params.hold_bars or close[i] >= regime[i]:
                        position = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                extreme = close[i]
                stop = close[i] - position * params.trail_atr * atr[i]
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    for side in (1, -1):
        close = 100.0 + side * np.array([0, 0, 2, -1, 0, 0], dtype=float)
        data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"ma200": 100.0 - side * 10, "atr": 1.0,
                            "z": side * np.array([-1.2, -0.8, -0.7, -0.6, -0.5, -0.4]),
                            "compression": 0.5, "wave": -1.0}, index=index)
        ind["z_previous"] = ind["z"].shift(1)
        result = strategy.generate_signals(data, ind, None, WaveParams()).data
        assert result["signal"].tolist() == [0, 0, side, side, 0, 0]
        ind["atr"] = 10.0
        result = strategy.generate_signals(data, ind, None, WaveParams()).data
        assert result["signal"].tolist() == [0, 0, side, side, side, side]


if __name__ == "__main__":
    _self_check()
