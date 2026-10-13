from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    area_jump: float = 0.12
    pressure_floor: float = 0.20


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local_1791322017"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return 4

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        if not np.isfinite(params.area_jump) or params.area_jump <= 0:
            raise ValueError("area_jump must be finite and positive")
        if not np.isfinite(params.pressure_floor) or not 0 <= params.pressure_floor < 1:
            raise ValueError("pressure_floor must be finite and in [0, 1)")
        previous_close = data["close"].shift(1)
        previous_close = previous_close.where(previous_close > 0)
        gap = (data["open"] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        body = ((data["close"] - data["open"]) / previous_close).replace([np.inf, -np.inf], np.nan)
        # Antisymmetric second-level signature of two consecutive (gap, body) increments.
        signed_area = 0.5 * (gap.shift(1) * body - body.shift(1) * gap)
        energy = (gap.pow(2) + body.pow(2)).rolling(2, min_periods=2).sum()
        area = signed_area / energy.clip(lower=1e-12)
        pressure = (body - gap) / (body.abs() + gap.abs()).clip(lower=1e-12)
        return pd.DataFrame({
            "gap": gap,
            "body": body,
            "signed_area": signed_area,
            "energy": energy,
            "area": area,
            "area_change": area.diff(),
            "pressure": pressure,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RotationParams) -> SignalFrame:
        jump = indicators["area_change"]
        pressure = indicators["pressure"]
        valid = jump.notna() & pressure.notna()
        long_entry = valid & (jump > params.area_jump) & (pressure > params.pressure_floor)
        short_entry = valid & (jump < -params.area_jump) & (pressure < -params.pressure_floor)
        requests = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, request in enumerate(requests):
            if position == 0:
                position = int(request)
            elif request == -position:
                # Opposite agreement exits to flat; neutral readings preserve the position.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = RotationParams()
    derived = strategy.indicators(data, params)
    assert derived["area_change"].iloc[:3].isna().all()
    assert derived["area_change"].iloc[3:].eq(0).all()
    probes = pd.DataFrame({"area_change": [0.2, np.nan, -0.2, -0.2, 0.2, 0.2],
                           "pressure": [0.5, np.nan, -0.5, -0.5, -0.5, 0.5]}, index=index)
    result = strategy.generate_signals(data, probes, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
