from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    range_bars: int = 12
    path_bars: int = 4
    area_threshold: float = 0.08
    pressure_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791235125"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(200, params.range_bars + params.path_bars)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.range_bars < 2 or params.path_bars < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.area_threshold < 0.5 or not 0.0 <= params.pressure_threshold < 1.0:
            raise ValueError("Invalid area or pressure threshold")
        low = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min()
        high = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max()
        x = (data["close"] - low) / (high - low).replace(0.0, np.nan)
        y = (2.0 * data["close"] - data["high"] - data["low"]) / (data["high"] - data["low"]).replace(0.0, np.nan)
        dx, dy = x.diff(), y.diff()
        x0, y0 = x.shift(params.path_bars), y.shift(params.path_bars)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-level signature of the translated (x, y) path.
        # Pandas window semantics: https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html
        # Shift semantics: https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html
        for lag in range(params.path_bars):
            area += (x.shift(lag + 1) - x0) * dy.shift(lag) - (y.shift(lag + 1) - y0) * dx.shift(lag)
        scale = dx.abs().rolling(params.path_bars, min_periods=params.path_bars).sum() * dy.abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        area = (0.5 * area / scale.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({
            "position": x,
            "pressure": y,
            "area": area,
            "ma200": data["close"].rolling(200, min_periods=200).mean(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        valid = np.isfinite(indicators).all(axis=1) & np.isfinite(data["close"])
        bullish = valid & (indicators["position"] < 0.5) & (indicators["pressure"] > params.pressure_threshold) & (indicators["area"] < -params.area_threshold)
        bearish = valid & (indicators["position"] > 0.5) & (indicators["pressure"] < -params.pressure_threshold) & (indicators["area"] > params.area_threshold)
        direction = np.select([bullish, bearish], [1, -1], default=0)
        entries = np.select([
            bullish & (data["close"] > indicators["ma200"]),
            bearish & (data["close"] < indicators["ma200"]),
        ], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        held = 0
        for i in range(len(data)):
            # The MA gates new entries; only an opposing setup exits a holding.
            if held and direction[i] == -held:
                held = 0
            elif held == 0:
                held = int(entries[i])
            signals[i] = held
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": 100.0}, index=index)
    sides = np.array([1, 0, -1, -1, 0, 1, 1, 0])
    indicators = pd.DataFrame({
        "position": np.where(sides == 1, 0.3, 0.7),
        "pressure": sides * 0.8,
        "area": sides * -0.2,
        "ma200": [90, 110, 90, 110, 110, 110, 90, 90],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
