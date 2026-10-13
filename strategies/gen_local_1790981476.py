from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverloadParams:
    vol_window: int = 20
    overload_multiple: float = 1.25
    min_streak: int = 2
    target_vol: float = 0.12
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OverloadParams]):
    strategy_id = "gen_local_1790981476"

    @classmethod
    def params_type(cls):
        return OverloadParams

    def warmup_bars(self, params: OverloadParams) -> int:
        return int(max(200, params.vol_window + 2))

    def indicators(self, data: pd.DataFrame, params: OverloadParams) -> pd.DataFrame:
        if params.vol_window < 2 or params.min_streak < 1:
            raise ValueError("vol_window must be >= 2 and min_streak >= 1")
        if not np.isfinite(params.overload_multiple) or params.overload_multiple <= 1:
            raise ValueError("overload_multiple must be finite and > 1")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["daily_vol"] = np.sqrt(out["return"].pow(2).rolling(params.vol_window, min_periods=params.vol_window).mean())
        out["capacity"] = out["daily_vol"].shift(1)
        out["overflow"] = (out["return"].abs() > params.overload_multiple * out["capacity"]).fillna(False)
        groups = (~out["overflow"]).cumsum()
        out["streak"] = out["overflow"].astype(int).groupby(groups).cumsum()
        out["pressure"] = out["return"].where(out["overflow"], 0.0).groupby(groups).cumsum()
        out["previous_streak"] = out["streak"].shift(1).fillna(0)
        out["previous_pressure"] = out["pressure"].shift(1).fillna(0.0)
        out["quiet"] = (out["return"].abs() <= out["capacity"]).fillna(False)
        out["ready"] = out[["return", "ma200", "daily_vol", "capacity"]].notna().all(axis=1) & (out["daily_vol"] > 0)
        out["release"] = out["ready"] & out["quiet"] & (out["previous_streak"] >= params.min_streak)
        long_entry = out["release"] & (close > out["ma200"]) & (out["previous_pressure"] < 0)
        short_entry = out["release"] & (close < out["ma200"]) & (out["previous_pressure"] > 0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["exposure"] = (params.target_vol / out["annual_vol"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(0.05, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OverloadParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        exposures = indicators["exposure"].fillna(0.05).clip(0.05, 0.95).to_numpy(dtype=float)
        position = 0
        exit_bar = -1
        entry_size = 0.95
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_size = float(exposures[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
            sizes[i] = entry_size if position else 0.95
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, -1, -1, 0, -1, 1, 1, 0], "exposure": [0.4, 0.2, 0.1, 0.5, 0.3, 0.6, 0.7, 0.8]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, OverloadParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert np.allclose(result.loc[index[[1, 2, 5, 6]], "size"], [0.4, 0.4, 0.3, 0.3])
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
