from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionAreaParams:
    area_threshold: float = 0.06
    profit_target: float = 0.018


class GeneratedStrategy(BaseStrategy[PositionAreaParams]):
    strategy_id = "gen_local_1791295712"

    @classmethod
    def params_type(cls) -> type[PositionAreaParams]:
        return PositionAreaParams

    def warmup_bars(self, params: PositionAreaParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: PositionAreaParams) -> pd.DataFrame:
        if not (0 < params.area_threshold < 0.5 and 0 < params.profit_target < 1):
            raise ValueError("area_threshold and profit_target must be positive")
        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(20, min_periods=20).max()
        out["channel_low"] = data["low"].rolling(20, min_periods=20).min()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["bar_width"] = data["high"] - data["low"]
        out["channel_position"] = ((data["close"] - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0)).clip(0, 1)
        out["bar_position"] = ((data["close"] - data["low"]) / out["bar_width"].where(out["bar_width"] > 0)).clip(0, 1)
        out["dx"] = out["channel_position"].diff()
        out["dy"] = out["bar_position"].diff()
        out["area"] = 0.5 * (out["dx"].shift(1) * out["dy"] - out["dy"].shift(1) * out["dx"])
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PositionAreaParams) -> SignalFrame:
        setup = ((indicators["area"] > params.area_threshold) & indicators["channel_position"].between(0.2, 0.6) & (indicators["bar_position"] > 0.6)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                if target_hit or i - entry_bar + 1 >= 4:
                    active = False
            elif i >= self.warmup_bars(params) and setup[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=30)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"area": 0.0, "channel_position": 0.4, "bar_position": 0.8}, index=index)
    indicators.loc[index[22], "area"] = 0.1
    indicators.loc[index[28], "area"] = 0.1
    data.loc[index[29], "close"] = 103.0
    result = GeneratedStrategy().generate_signals(data, indicators, None, PositionAreaParams())
    assert result.data["signal"].iloc[22:30].tolist() == [0, 1, 1, 1, 1, 0, 0, 1]
    assert result.data["signal"].isin([0, 1]).all()


if __name__ == "__main__":
    _self_check()
