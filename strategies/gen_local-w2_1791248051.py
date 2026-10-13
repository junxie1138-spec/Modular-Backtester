from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    signature_window: int = 6
    pressure_window: int = 3
    vol_window: int = 20
    area_threshold: float = 0.08
    pressure_threshold: float = 0.15
    holding_bars: int = 4
    target_vol: float = 0.10
    vol_floor: float = 0.01
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791248051"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.signature_window + 1, params.pressure_window, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.signature_window, params.pressure_window, params.vol_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")
        if not (0 < params.target_vol and 0 < params.vol_floor and 0 < params.max_size <= 1):
            raise ValueError("Volatility targets and exposure bounds must be positive.")
        if not (0 <= params.area_threshold <= 1 and 0 <= params.pressure_threshold <= 1):
            raise ValueError("Thresholds must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["activity_change"] = np.log(data["volume"].where(data["volume"] > 0)).diff()
        r, v = out["return"], out["activity_change"]
        window = params.signature_window
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature: each earlier increment crossed with each later increment.
        for lag in range(1, window):
            area = area + (r.shift(lag) * v - v.shift(lag) * r).rolling(window - lag).sum()
        out["signature_area"] = 0.5 * area
        out["area_scale"] = 0.5 * r.abs().rolling(window).sum() * v.abs().rolling(window).sum()
        out["area_score"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        spread = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) / spread).clip(-1, 1)
        volume = data["volume"].where(data["volume"] > 0)
        out["pressure_volume"] = out["close_location"] * volume
        out["pressure_sum"] = out["pressure_volume"].rolling(params.pressure_window).sum()
        out["volume_sum"] = volume.rolling(params.pressure_window).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["annual_vol"] = r.rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].clip(lower=params.vol_floor)).clip(upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        area, pressure = indicators["area_score"], indicators["pressure"]
        ready = indicators[["area_score", "pressure", "target_size"]].notna().all(axis=1)
        ready &= indicators["target_size"] > 0
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        long_entry = ready & (area > params.area_threshold) & (pressure > params.pressure_threshold)
        short_entry = ready & (area < -params.area_threshold) & (pressure < -params.pressure_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.max_size, dtype=float)
        position, remaining, exposure = 0, 0, params.max_size
        for i, candidate in enumerate(candidates):
            if position:
                if remaining == 0:
                    position, exposure = 0, params.max_size
                else:
                    remaining -= 1
            elif candidate:
                position = int(candidate)
                remaining = params.holding_bars - 1
                exposure = float(indicators["target_size"].iloc[i])
            signal[i], sizes[i] = position, exposure
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    p = Params(signature_window=2, pressure_window=2, vol_window=2)
    index = pd.date_range("2020-01-01", periods=16)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0,
                         "close": 101.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"area_score": 0.5, "pressure": 0.5, "target_size": 0.4}, index=index)
    ind.loc[index[3]:, ["area_score", "pressure"]] = -0.5
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, StrategyContext("SPY", "1d", 3), p).data
    assert result["signal"].tolist()[:13] == [0, 0, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert (result.loc[result["signal"] != 0, "size"] == 0.4).all()
    ind["pressure"] = -ind["area_score"]
    assert (strategy.generate_signals(data, ind, StrategyContext("SPY", "1d", 3), p).data["signal"] == 0).all()
    assert strategy.indicators(data, p)["area_score"].isna().all()


if __name__ == "__main__":
    _self_check()
