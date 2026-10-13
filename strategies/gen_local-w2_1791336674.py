from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 20
    inventory_window: int = 8
    drawdown_floor: float = 0.015
    volume_dominance: float = 0.60


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791336674"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.peak_window + params.inventory_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.peak_window < 2 or params.inventory_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.drawdown_floor < 1.0:
            raise ValueError("drawdown_floor must be between zero and one")
        if not 0.5 < params.volume_dominance < 1.0:
            raise ValueError("volume_dominance must be between 0.5 and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0)
        out["prior_peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        out["drawdown"] = 1.0 - close.div(out["prior_peak"].where(out["prior_peak"] > 0))
        out["change"] = close.diff()
        out["underwater"] = out["drawdown"].ge(params.drawdown_floor)
        valid = out["prior_peak"].notna() & close.notna() & volume.notna() & out["change"].notna()
        out["inventory_volume"] = volume.where(out["underwater"], 0.0).where(valid)
        out["inventory_value"] = (close * out["inventory_volume"]).where(valid)
        out["inventory_volume_sum"] = out["inventory_volume"].rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        out["inventory_value_sum"] = out["inventory_value"].rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        out["inventory_basis"] = out["inventory_value_sum"].div(out["inventory_volume_sum"].where(out["inventory_volume_sum"] > 0))
        out["up_volume"] = out["inventory_volume"].where(out["change"] > 0, 0.0).where(valid)
        out["down_volume"] = out["inventory_volume"].where(out["change"] < 0, 0.0).where(valid)
        out["up_volume_sum"] = out["up_volume"].rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        out["down_volume_sum"] = out["down_volume"].rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        out["directional_volume"] = out["up_volume_sum"] + out["down_volume_sum"]
        out["buy_share"] = out["up_volume_sum"].div(out["directional_volume"].where(out["directional_volume"] > 0))
        out["ready"] = valid & out["inventory_basis"].notna() & out["buy_share"].notna()
        out["long_condition"] = out["ready"] & out["underwater"] & close.gt(out["inventory_basis"]) & out["buy_share"].ge(params.volume_dominance)
        out["short_condition"] = out["ready"] & out["underwater"] & close.lt(out["inventory_basis"]) & out["buy_share"].le(1.0 - params.volume_dominance)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        directions = np.select(
            [indicators["long_condition"].fillna(False).to_numpy(dtype=bool),
             indicators["short_condition"].fillna(False).to_numpy(dtype=bool)],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, direction in enumerate(directions):
            if position != 0 and direction == -position:
                position = 0
            elif position == 0 and direction != 0 and direction == previous:
                position = int(direction)
            signals[i] = position
            previous = int(direction)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    directions = np.array([1, 1, 0, -1, -1, -1, 1, 1])
    indicators = pd.DataFrame({"long_condition": directions == 1, "short_condition": directions == -1}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()
    empty = strategy.generate_signals(data.iloc[:0], indicators.iloc[:0], None, Params())
    assert empty.data.empty


if __name__ == "__main__":
    _self_check()
