from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolumeParams:
    history_bars: int = 378
    horizon_bars: int = 18
    volume_window: int = 20
    volume_multiple: float = 1.15
    minimum_events: int = 6
    forecast_threshold: float = 0.01


class GeneratedStrategy(BaseStrategy[SeasonalVolumeParams]):
    strategy_id = "gen_local_1790910633"

    @classmethod
    def params_type(cls):
        return SeasonalVolumeParams

    def warmup_bars(self, params: SeasonalVolumeParams) -> int:
        return int(params.history_bars + params.horizon_bars + params.volume_window + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalVolumeParams) -> pd.DataFrame:
        if min(params.history_bars, params.horizon_bars, params.volume_window, params.minimum_events) < 1:
            raise ValueError("Lookbacks and minimum_events must be positive")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be finite and positive")
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError("forecast_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = volume.rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out["relative_volume"] = (volume / out["volume_baseline"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["return"]).where(out["relative_volume"] >= params.volume_multiple, 0).fillna(0).astype(int)
        out["month_half"] = (data.index.day > 15).astype(int)
        out["matured_return"] = (close / close.shift(params.horizon_bars) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["forecast"] = np.nan
        for half in (0, 1):
            for direction in (-1, 1):
                name = f"half_{half}_{'up' if direction == 1 else 'down'}"
                current = (out["month_half"] == half) & (out["direction"] == direction)
                matured = current.shift(params.horizon_bars, fill_value=False) & out["matured_return"].notna()
                out[f"{name}_matured_event"] = matured.astype(int)
                out[f"{name}_outcome"] = out["matured_return"].where(matured, 0.0)
                out[f"{name}_count"] = out[f"{name}_matured_event"].rolling(params.history_bars, min_periods=params.history_bars).sum()
                out[f"{name}_total"] = out[f"{name}_outcome"].rolling(params.history_bars, min_periods=params.history_bars).sum()
                count = out[f"{name}_count"]
                out[f"{name}_forecast"] = (out[f"{name}_total"] / count.replace(0.0, np.nan)).where(count >= params.minimum_events)
                out.loc[current, "forecast"] = out.loc[current, f"{name}_forecast"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalVolumeParams) -> SignalFrame:
        forecast = indicators["forecast"].replace([np.inf, -np.inf], np.nan)
        state = pd.Series(np.nan, index=data.index, dtype=float)
        state.loc[forecast >= params.forecast_threshold] = 1.0
        state.loc[forecast <= -params.forecast_threshold] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = state.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-02", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"forecast": [0.02, 0.0, -0.02, np.nan, 0.02]}, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalVolumeParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["forecast"].isna().all()


if __name__ == "__main__":
    _self_check()
