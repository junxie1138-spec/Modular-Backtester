from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAreaParams:
    lookback: int = 5
    area_threshold: float = 0.20
    profit_target: float = 0.009
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[SeasonalAreaParams]):
    strategy_id = "gen_local_1791256934"

    @classmethod
    def params_type(cls) -> type[SeasonalAreaParams]:
        return SeasonalAreaParams

    @staticmethod
    def warmup_bars(params: SeasonalAreaParams) -> int:
        return max(200, max(2, int(params.lookback)) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalAreaParams) -> pd.DataFrame:
        length = max(2, int(params.lookback))
        close = data["close"].astype(float)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        clock = np.cos(2.0 * np.pi * weekday / 5.0)
        clock_step = clock.diff()
        price_step = np.log(close.where(close > 0)).diff()
        area = pd.Series(0.0, index=data.index)
        for older in range(1, length):
            for newer in range(older):
                area = area + 0.5 * (
                    clock_step.shift(older) * price_step.shift(newer)
                    - price_step.shift(older) * clock_step.shift(newer)
                )
        scale = (
            clock_step.abs().rolling(length, min_periods=length).sum()
            * price_step.abs().rolling(length, min_periods=length).sum()
        )
        normalized_area = (area / scale.where(scale > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        return pd.DataFrame({
            "weekday": weekday,
            "area": normalized_area,
            "ma200": close.rolling(200, min_periods=200).mean(),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalAreaParams,
    ) -> SignalFrame:
        if not np.isfinite(params.area_threshold) or params.area_threshold < 0:
            raise ValueError("area_threshold must be finite and nonnegative")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        area = indicators["area"].to_numpy(dtype=float)
        average = indicators["ma200"].to_numpy(dtype=float)
        weekday = indicators["weekday"].to_numpy(dtype=int)
        valid = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(area) & np.isfinite(average)
        )
        candidates = np.zeros(len(data), dtype=int)
        candidates[valid & (weekday <= 1) & (close > average)
                   & (area >= params.area_threshold)] = 1
        candidates[valid & (weekday >= 3) & (weekday <= 4)
                   & (close < average)
                   & (area <= -params.area_threshold)] = -1
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError("Position entry requires a valid positive open")
                    entry_price = opens[i]
                held = i - entry_bar + 1
                profit = (
                    position * (close[i] / entry_price - 1.0)
                    if np.isfinite(close[i]) and close[i] > 0 else -np.inf
                )
                if profit >= params.profit_target or held >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
            elif candidates[i]:
                position = int(candidates[i])
                entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SeasonalAreaParams()
    index = pd.bdate_range("2020-01-06", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [103.0] * 8,
        "low": [98.0] * 8,
        "close": [101.0, 102.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000000.0] * 8,
    }, index=index)
    features = pd.DataFrame({
        "weekday": index.dayofweek,
        "area": [0.8, 0.0, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0],
        "ma200": [100.0] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0, 0]
    assert (result.data["size"] > 0).all()
    warmup_features = GeneratedStrategy.indicators(data, params)
    assert warmup_features.index.equals(data.index)
    assert warmup_features["ma200"].isna().all()
    empty_result = GeneratedStrategy.generate_signals(data, warmup_features, None, params)
    assert empty_result.data["signal"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200
