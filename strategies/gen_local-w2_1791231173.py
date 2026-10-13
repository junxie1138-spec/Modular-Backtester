from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoolingFadeParams:
    ma_bars: int = 10
    vol_bars: int = 10
    peak_bars: int = 5
    entry_z: float = 1.0
    cooling_ratio: float = 0.9
    target_vol: float = 0.10


class GeneratedStrategy(BaseStrategy[CoolingFadeParams]):
    strategy_id = "gen_local-w2_1791231173"

    @classmethod
    def params_type(cls) -> type[CoolingFadeParams]:
        return CoolingFadeParams

    def warmup_bars(self, params: CoolingFadeParams) -> int:
        return int(max(params.ma_bars, params.vol_bars + params.peak_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: CoolingFadeParams) -> pd.DataFrame:
        if min(params.ma_bars, params.vol_bars) < 2 or params.peak_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not (np.isfinite(params.entry_z) and params.entry_z > 0):
            raise ValueError("entry_z must be finite and positive")
        if not (0 < params.cooling_ratio < 1):
            raise ValueError("cooling_ratio must be between zero and one")
        if not (np.isfinite(params.target_vol) and params.target_vol > 0):
            raise ValueError("target_vol must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["price_std"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["vol"] = out["return"].rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0) * np.sqrt(252.0)
        out["vol_peak"] = out["vol"].rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out["cooling"] = out["vol"].gt(0) & out["vol"].le(params.cooling_ratio * out["vol_peak"])
        out["direction"] = np.select(
            [out["cooling"] & out["z"].le(-params.entry_z),
             out["cooling"] & out["z"].ge(params.entry_z)],
            [1, -1], default=0,
        ).astype(int)
        out["confirmed"] = out["direction"].where(out["direction"].eq(out["direction"].shift(1)), 0).astype(int)
        out["exposure"] = (params.target_vol / out["vol"].clip(lower=0.03)).clip(lower=0.05, upper=0.95).fillna(0.05)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CoolingFadeParams) -> SignalFrame:
        position = 0
        states = np.zeros(len(data), dtype=int)
        for i, (direction, entry) in enumerate(zip(indicators["direction"], indicators["confirmed"])):
            if position and direction == -position:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
            states[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = states
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CoolingFadeParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame(index=index)
    features["direction"] = [1, 1, 0, -1, -1, -1, 0, 1, 1]
    features["confirmed"] = features["direction"].where(features["direction"].eq(features["direction"].shift(1)), 0).astype(int)
    features["exposure"] = np.linspace(0.1, 0.9, len(index))
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, -1, 0]
    assert np.allclose(result["size"], features["exposure"].shift(1).fillna(0.05))
    flat = pd.concat([data] * 5, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat))
    flat_features = strategy.indicators(flat, params)
    flat_result = strategy.generate_signals(flat, flat_features, None, params).data
    assert flat_result["signal"].eq(0).all()
    assert np.isfinite(flat_result["size"]).all() and flat_result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.vol_bars + params.peak_bars + 1


if __name__ == "__main__":
    _self_check()
