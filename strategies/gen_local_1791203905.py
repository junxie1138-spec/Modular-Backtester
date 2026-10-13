from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReclaimParams:
    range_window: int = 6
    acceleration_floor: float = 0.002
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ReclaimParams]):
    strategy_id = "gen_local_1791203905"

    @classmethod
    def params_type(cls) -> type[ReclaimParams]:
        return ReclaimParams

    def warmup_bars(self, params: ReclaimParams) -> int:
        if not 3 <= params.range_window <= 8:
            raise ValueError("range_window must be between 3 and 8")
        return int(params.range_window + 2)

    def indicators(self, data: pd.DataFrame, params: ReclaimParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["range_high"] = data["high"].rolling(params.range_window).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_window).min().shift(1)
        width = out["range_high"] - out["range_low"]
        out["range_width"] = width
        out["position"] = (close - out["range_low"]) / width.where(width > 0)
        out["previous_position"] = out["position"].shift(1)
        out["roc"] = close.pct_change(periods=2, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["previous_position"] < 0.5)
            & (out["position"] >= 0.5)
            & (out["position"] < 1.0)
            & (out["roc"] < 0.0)
            & (out["acceleration"] > params.acceleration_floor)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReclaimParams,
    ) -> SignalFrame:
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not np.isfinite(params.acceleration_floor) or params.acceleration_floor < 0:
            raise ValueError("acceleration_floor must be finite and nonnegative")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_stop:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                # The shifted entry fills at the next bar's open.
                entry_bar = i + 1
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True, True, True, False, False]}, index=index)
    params = ReclaimParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    indicators["entry"] = [True, False, False, False, False]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    params.max_hold_bars = 1
    data["close"] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(None, data, params)
    assert not derived["entry"].any()
    assert GeneratedStrategy.warmup_bars(None, params) <= 10
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
