from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    ma_window: int = 20
    z_entry: float = 1.25
    min_gap: float = 0.003
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1791233903"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return max(params.ma_window, 2)

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["ma"] = data["close"].rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = data["close"].rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (data["close"] - out["ma"]) / out["std"].where(out["std"] > 0)
        out["gap"] = data["open"] / out["previous_close"].where(out["previous_close"] > 0) - 1.0
        gap_depth = out["previous_close"] - data["open"]
        out["gap_repair"] = (data["close"] - data["open"]) / gap_depth.where(gap_depth > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        entries = (
            (indicators["z"] <= -params.z_entry)
            & (indicators["gap"] <= -params.min_gap)
            & (indicators["gap_repair"] >= 1.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # A preceding close decision first becomes a position at this open.
            if i > 0 and signal[i - 1] == 1 and entry_bar < 0:
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                target_hit = closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapRepairParams(max_hold=3)
    index = pd.date_range("2020-01-01", periods=8)
    close = [100.0, 102.6, 104.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    data = pd.DataFrame(
        {"open": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
         "high": np.maximum(close, 101.0), "low": 99.0, "close": close, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"z": [-2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -2.0],
         "gap": [-0.01, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0],
         "gap_repair": 1.0}, index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    cold = strategy.indicators(data, params)
    assert strategy.generate_signals(data, cold, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
