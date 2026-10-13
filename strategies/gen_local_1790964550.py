from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 6
    area_threshold: float = 0.12
    min_efficiency: float = 0.35
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790964550"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        if not 2 <= params.window <= 9:
            raise ValueError("window must be between 2 and 9")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        if not 0.0 < params.area_threshold < 0.5:
            raise ValueError("area_threshold must be between 0 and 0.5")
        if not 0.0 < params.min_efficiency <= 1.0:
            raise ValueError("min_efficiency must be between 0 and 1")
        return int(params.window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        result["price_step"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        result["turnover_step"] = data["volume"].astype(float).where(data["volume"] >= 0).replace([np.inf, -np.inf], np.nan)
        prefix_price = pd.Series(0.0, index=data.index)
        prefix_turnover = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        for offset in range(params.window - 1, -1, -1):
            price_step = result["price_step"].shift(offset)
            turnover_step = result["turnover_step"].shift(offset)
            area = area + 0.5 * (prefix_price * turnover_step - prefix_turnover * price_step)
            prefix_price = prefix_price + price_step
            prefix_turnover = prefix_turnover + turnover_step
        result["signature_area"] = area
        result["displacement"] = prefix_price
        result["turnover"] = prefix_turnover
        result["path_length"] = result["price_step"].abs().rolling(params.window, min_periods=params.window).sum()
        denominator = result["path_length"] * result["turnover"]
        result["normalized_area"] = result["signature_area"].div(denominator.where(denominator > 0))
        result["efficiency"] = result["displacement"].abs().div(result["path_length"].where(result["path_length"] > 0))
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        eligible = indicators["efficiency"].ge(params.min_efficiency)
        long_entry = eligible & indicators["displacement"].lt(0) & indicators["normalized_area"].le(-params.area_threshold)
        short_entry = eligible & indicators["displacement"].gt(0) & indicators["normalized_area"].ge(params.area_threshold)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar == params.hold_bars:
                    position = 0
            elif entries[bar] != 0:
                position = int(entries[bar])
                entry_bar = bar
            positions[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    close = np.exp(np.cumsum(np.r_[0.0, np.full(23, 0.01)]))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": np.ones(24)}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.warmup_bars(params) <= 10
    assert np.allclose(derived["normalized_area"].dropna(), 0.0, atol=1e-12)
    derived["displacement"] = 0.1
    derived["efficiency"] = 1.0
    derived["normalized_area"] = 0.3
    signals = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].iloc[0] == 0
    assert signals["signal"].iloc[1:9].eq(-1).all()
    assert signals["signal"].iloc[9] == 0
    derived["displacement"] = -0.1
    derived["normalized_area"] = -0.3
    signals = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].iloc[1:9].eq(1).all()
    assert signals["signal"].iloc[9] == 0
