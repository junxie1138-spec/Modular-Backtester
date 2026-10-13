from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityAllocationParams:
    vol_window: int = 20
    dominance: float = 0.60
    body_threshold: float = 0.80
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VolatilityAllocationParams]):
    strategy_id = "gen_local_1791264291"

    @classmethod
    def params_type(cls):
        return VolatilityAllocationParams

    def warmup_bars(self, params: VolatilityAllocationParams) -> int:
        return max(200, params.vol_window + 1)

    def indicators(self, data: pd.DataFrame, params: VolatilityAllocationParams) -> pd.DataFrame:
        if params.vol_window < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Require vol_window >= 2 and hold_bars between 3 and 5.")
        if not 0.5 < params.dominance < 1.0 or not np.isfinite(params.body_threshold) or params.body_threshold <= 0:
            raise ValueError("Require 0.5 < dominance < 1 and a positive finite body_threshold.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["overnight_return"] = np.log(opening / close.shift(1))
        out["intraday_return"] = np.log(close / opening)
        out["overnight_std"] = out["overnight_return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out["intraday_std"] = out["intraday_return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out["total_variance"] = out["overnight_std"].pow(2) + out["intraday_std"].pow(2)
        out["overnight_share"] = out["overnight_std"].pow(2) / out["total_variance"].where(out["total_variance"] > 0)
        out["intraday_z"] = out["intraday_return"] / out["intraday_std"].where(out["intraday_std"] > 0)
        out["regime"] = np.where(close > out["ma200"], 1, np.where(close < out["ma200"], -1, 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityAllocationParams) -> SignalFrame:
        aligned_body = indicators["intraday_z"] * indicators["regime"]
        continuation = (indicators["overnight_share"] <= 1.0 - params.dominance) & (aligned_body >= params.body_threshold)
        reversal = (indicators["overnight_share"] >= params.dominance) & (aligned_body <= -params.body_threshold)
        valid = indicators[["ma200", "overnight_share", "intraday_z", "regime"]].notna().all(axis=1)
        candidates = indicators["regime"].where(valid & (continuation | reversal), 0).fillna(0).astype(int).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VolatilityAllocationParams()
    index = pd.date_range("2020-01-01", periods=16, freq="B")
    data = pd.DataFrame(index=index)
    regime = np.array([1] * 8 + [-1] * 8)
    indicators = pd.DataFrame({
        "ma200": 100.0,
        "overnight_share": 0.7,
        "intraday_z": -regime.astype(float),
        "regime": regime,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    indicators["overnight_share"] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
