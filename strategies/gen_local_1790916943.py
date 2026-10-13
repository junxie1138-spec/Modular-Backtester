from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticRegimeParams:
    ma_bars: int = 60
    regime_bars: int = 40
    pressure_bars: int = 5
    z_threshold: float = 1.25
    efficiency_threshold: float = 0.30
    pressure_threshold: float = 0.08
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ElasticRegimeParams]):
    strategy_id = "gen_local_1790916943"

    @classmethod
    def params_type(cls) -> type[ElasticRegimeParams]:
        return ElasticRegimeParams

    def warmup_bars(self, params: ElasticRegimeParams) -> int:
        if min(params.ma_bars, params.regime_bars, params.pressure_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive")
        if not np.isfinite(params.efficiency_threshold) or not 0 < params.efficiency_threshold < 1:
            raise ValueError("efficiency_threshold must lie strictly between zero and one")
        if not np.isfinite(params.pressure_threshold) or not 0 < params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must lie strictly between zero and one")
        return int(max(params.ma_bars, params.regime_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: ElasticRegimeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["scale"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"] / out["scale"].replace(0.0, np.nan)
        out["net_change"] = close.diff(params.regime_bars).abs()
        out["path_length"] = close.diff().abs().rolling(params.regime_bars, min_periods=params.regime_bars).sum()
        out["efficiency"] = out["net_change"] / out["path_length"].replace(0.0, np.nan)
        out["bar_range"] = data["high"] - data["low"]
        out["closing_location"] = ((2.0 * close - data["high"] - data["low"]) / out["bar_range"].replace(0.0, np.nan)).where(out["bar_range"] != 0.0, 0.0).clip(-1.0, 1.0)
        out["pressure_volume"] = out["closing_location"] * data["volume"]
        out["pressure_sum"] = out["pressure_volume"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["volume_sum"] = data["volume"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure"] = out["pressure_sum"] / out["volume_sum"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticRegimeParams) -> SignalFrame:
        self.warmup_bars(params)
        valid = indicators[["z", "efficiency", "pressure"]].notna().all(axis=1)
        trending = indicators["efficiency"] >= params.efficiency_threshold
        z = indicators["z"]
        pressure = indicators["pressure"]
        long_stretch = (trending & (z >= params.z_threshold)) | (~trending & (z <= -params.z_threshold))
        short_stretch = (trending & (z <= -params.z_threshold)) | (~trending & (z >= params.z_threshold))
        entries = np.select(
            [valid & long_stretch & (pressure >= params.pressure_threshold),
             valid & short_stretch & (pressure <= -params.pressure_threshold)],
            [1, -1], default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar == params.hold_bars:
                    position = 0
                signals[bar] = position
                continue
            if entries[bar] != 0:
                position = int(entries[bar])
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ElasticRegimeParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    sample = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(sample, params)
    assert strategy.warmup_bars(params) >= 60
    assert derived.index.equals(sample.index)
    assert not np.isinf(derived.to_numpy()).any()
    synthetic = pd.DataFrame({"z": 2.0, "efficiency": 0.8, "pressure": 0.5}, index=index)
    synthetic.loc[index[1:19], ["z", "pressure"]] = [-2.0, -0.5]
    result = strategy.generate_signals(sample, synthetic, None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:19] == 1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20] == 1
    synthetic.loc[:, "efficiency"] = 0.1
    reversed_result = strategy.generate_signals(sample, synthetic, None, params).data
    assert reversed_result["signal"].eq(0).all()
    synthetic.loc[:, "pressure"] = -0.5
    restoring_result = strategy.generate_signals(sample, synthetic, None, params).data
    assert (restoring_result["signal"].iloc[1:19] == -1).all()
    assert restoring_result["signal"].iloc[19] == 0
