from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalSpringParams:
    ma_window: int = 10
    entry_z: float = 0.8
    seasonal_tilt: float = 0.45


class GeneratedStrategy(BaseStrategy[SeasonalSpringParams]):
    strategy_id = "gen_local-w2_1791327404"

    @classmethod
    def params_type(cls) -> type[SeasonalSpringParams]:
        return SeasonalSpringParams

    def warmup_bars(self, params: SeasonalSpringParams) -> int:
        return int(params.ma_window) + 1

    def indicators(self, data: pd.DataFrame, params: SeasonalSpringParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.entry_z <= 0 or params.seasonal_tilt < 0:
            raise ValueError("Require ma_window >= 2, entry_z > 0, seasonal_tilt >= 0")
        close = data["close"].astype(float)
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        scale = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        z = ((close - mean) / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        phase = 2.0 * np.pi * (data.index.day.to_numpy() - 1) / data.index.days_in_month.to_numpy()
        anchor = pd.Series(params.seasonal_tilt * np.cos(phase), index=data.index)
        return pd.DataFrame({"mean": mean, "scale": scale, "z": z, "seasonal_anchor": anchor, "tension": z - anchor}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalSpringParams) -> SignalFrame:
        tension = indicators["tension"]
        valid = tension.notna() & np.isfinite(tension)
        candidate = pd.Series(np.select([valid & (tension <= -params.entry_z), valid & (tension >= params.entry_z)], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.ne(0) & candidate.eq(candidate.shift(1)), 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(confirmed.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"tension": [-2.0, -2.0, np.nan, 2.0, 2.0, 2.0, -2.0, -2.0]}, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalSpringParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["tension"].isna().all()
    assert strategy.warmup_bars(params) >= params.ma_window + 1


if __name__ == "__main__":
    _self_check()
