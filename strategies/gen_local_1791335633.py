from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 8
    seasonal_window: int = 126
    entry_threshold: float = 1.75
    hold_bars: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791335633"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.slope_window + params.seasonal_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.slope_window < 2 or params.seasonal_window < 30:
            raise ValueError("Insufficient slope or seasonal lookback")
        if params.entry_threshold <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("Threshold must be positive and holding period must be one or two bars")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Position sizes must satisfy 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["weekday"] = data.index.dayofweek
        length = params.slope_window
        out["log_sum"] = out["log_close"].rolling(length, min_periods=length).sum()
        out["time_mean"] = out["bar_number"].rolling(length, min_periods=length).mean()
        out["cross_sum"] = (out["log_close"] * out["bar_number"]).rolling(length, min_periods=length).sum()
        denominator = length * (length * length - 1) / 12.0
        out["slope"] = (out["cross_sum"] - out["time_mean"] * out["log_sum"]) / denominator
        out["innovation"] = out["slope"].diff()
        out["seasonal_mean"] = np.nan
        out["seasonal_std"] = np.nan
        out["seasonal_count"] = 0.0
        window = params.seasonal_window
        for weekday in range(5):
            mask = out["weekday"].eq(weekday)
            valid = mask & out["innovation"].notna()
            previous = out["innovation"].where(valid, 0.0).shift(1)
            count = valid.astype(float).shift(1).rolling(window, min_periods=window).sum()
            total = previous.rolling(window, min_periods=window).sum()
            squares = previous.pow(2).rolling(window, min_periods=window).sum()
            mean = total / count.where(count >= 4)
            variance = (squares - total * mean) / (count - 1).where(count >= 4)
            out.loc[mask, "seasonal_count"] = count.loc[mask]
            out.loc[mask, "seasonal_mean"] = mean.loc[mask]
            out.loc[mask, "seasonal_std"] = variance.clip(lower=0).pow(0.5).loc[mask]
        out["ready"] = out["seasonal_std"].gt(1e-12) & out["innovation"].notna()
        out["surprise"] = ((out["innovation"] - out["seasonal_mean"]) / out["seasonal_std"].where(out["ready"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        surprises = indicators["surprise"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_bar = -1
        entry_size = params.min_size
        first_bar = params.slope_window + params.seasonal_window
        for i in range(len(data)):
            if position:
                if i - entry_bar >= params.hold_bars:
                    position = 0
                    continue
                signals[i] = position
                sizes[i] = entry_size
                continue
            if i < first_bar or not ready[i] or not np.isfinite(surprises[i]):
                continue
            surprise = surprises[i]
            if abs(surprise) >= params.entry_threshold:
                position = -1 if surprise > 0 else 1
                entry_bar = i
                entry_size = float(np.clip(params.max_size * abs(surprise) / (2.0 * params.entry_threshold), params.min_size, params.max_size))
                signals[i] = position
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StrategyParams()
    entry = params.slope_window + params.seasonal_window
    index = pd.bdate_range("2020-01-01", periods=entry + 9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"surprise": 0.0, "ready": True}, index=index)
    indicators.iloc[entry, 0] = params.entry_threshold
    indicators.iloc[entry + 1, 0] = -3.0 * params.entry_threshold
    indicators.iloc[entry + 2, 0] = -3.0 * params.entry_threshold
    indicators.iloc[entry + 3, 0] = -3.0 * params.entry_threshold
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].iloc[entry:entry + 7].tolist() == [0, -1, -1, 0, 1, 1, 0]
    assert result["size"].iloc[entry + 1] == result["size"].iloc[entry + 2]
    assert result["size"].iloc[entry + 4] > result["size"].iloc[entry + 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
