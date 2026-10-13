from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeBalanceParams:
    channel_window: int = 15
    balance_window: int = 8
    compression_ratio: float = 1.2


class GeneratedStrategy(BaseStrategy[RangeBalanceParams]):
    strategy_id = "gen_local_1791324202"

    @classmethod
    def params_type(cls) -> type[RangeBalanceParams]:
        return RangeBalanceParams

    def warmup_bars(self, params: RangeBalanceParams) -> int:
        return int(params.channel_window + params.balance_window + 1)

    def indicators(self, data: pd.DataFrame, params: RangeBalanceParams) -> pd.DataFrame:
        if params.channel_window < 2 or params.balance_window < 4:
            raise ValueError("Channel window must be >= 2 and balance window >= 4")
        if not np.isfinite(params.compression_ratio) or params.compression_ratio <= 0:
            raise ValueError("Compression ratio must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["range_midpoint"] = (data["high"] + data["low"]) / 2.0
        out["channel_high"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max()
        out["channel_low"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min()
        out["channel_width"] = (out["channel_high"] - out["channel_low"]).where(lambda s: s > 0)
        out["channel_midpoint"] = (out["channel_high"] + out["channel_low"]) / 2.0
        out["channel_position"] = (data["close"] - out["channel_low"]) / out["channel_width"]
        valid = out["daily_range"].notna() & out["channel_width"].notna()
        out["upper_member"] = (out["range_midpoint"] >= out["channel_midpoint"]).astype(float).where(valid)
        out["lower_member"] = (1.0 - out["upper_member"]).where(valid)
        window = params.balance_window
        out["upper_count"] = out["upper_member"].rolling(window, min_periods=window).sum()
        out["lower_count"] = out["lower_member"].rolling(window, min_periods=window).sum()
        out["upper_range_sum"] = (out["daily_range"] * out["upper_member"]).rolling(window, min_periods=window).sum()
        out["lower_range_sum"] = (out["daily_range"] * out["lower_member"]).rolling(window, min_periods=window).sum()
        out["upper_range_mean"] = out["upper_range_sum"] / out["upper_count"].where(out["upper_count"] >= 2)
        out["lower_range_mean"] = out["lower_range_sum"] / out["lower_count"].where(out["lower_count"] >= 2)
        out["range_ratio"] = out["lower_range_mean"] / out["upper_range_mean"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeBalanceParams) -> SignalFrame:
        bullish = (indicators["range_ratio"].ge(params.compression_ratio) & indicators["channel_position"].ge(0.6)).fillna(False)
        # Two consecutive bullish closes enter; the first false close exits.
        confirmed = bullish & bullish.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeBalanceParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"range_ratio": [np.nan, 1.3, 1.3, 1.3, 0.9, 1.3, 1.3, 1.3], "channel_position": 0.75}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
