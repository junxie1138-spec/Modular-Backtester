from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    ma_bars: int = 60
    pressure_bars: int = 10
    entry_z: float = 1.25
    pressure_threshold: float = 0.12
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local-w2_1791256528"

    @classmethod
    def params_type(cls) -> type[AbsorptionParams]:
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        if params.ma_bars < 2 or params.pressure_bars < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; ma_bars must exceed one")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(params.pressure_threshold) or not 0 < params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must lie strictly between zero and one")
        return int(max(params.ma_bars, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma"] = data["close"].rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["price_std"] = data["close"].rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = (data["close"] - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["daily_range"] = data["high"] - data["low"]
        out["close_location"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["daily_range"].where(out["daily_range"] > 0)).clip(-1.0, 1.0)
        out.loc[out["daily_range"].eq(0), "close_location"] = 0.0
        out["valid_volume"] = data["volume"].where(data["volume"].ge(0))
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure"] = out["signed_volume_sum"] / out["volume_sum"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorptionParams) -> SignalFrame:
        z = indicators["z"].to_numpy(dtype=float)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        valid = np.isfinite(z) & np.isfinite(pressure)
        longs = valid & (z <= -params.entry_z) & (pressure >= params.pressure_threshold)
        shorts = valid & (z >= params.entry_z) & (pressure <= -params.pressure_threshold)
        entries = np.where(longs, 1, np.where(shorts, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = AbsorptionParams()
    index = pd.date_range("2020-01-01", periods=42, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"z": 0.0, "pressure": 0.0}, index=index)
    derived.iloc[0] = [-2.0, 0.3]
    derived.iloc[1:19] = [2.0, -0.3]
    derived.iloc[19] = [2.0, -0.3]
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    expected = np.zeros(42, dtype=int)
    expected[1:19] = 1
    expected[20:38] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(None, data, AbsorptionParams(ma_bars=5))
    assert flat["z"].isna().all()
