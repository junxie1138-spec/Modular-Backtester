from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DefenseParams:
    ma_window: int = 6
    defense_window: int = 4
    required_defenses: int = 2
    z_entry: float = 0.65


class GeneratedStrategy(BaseStrategy[DefenseParams]):
    strategy_id = "gen_local_1790966803"

    @classmethod
    def params_type(cls):
        return DefenseParams

    def warmup_bars(self, params: DefenseParams) -> int:
        bars = params.ma_window + params.defense_window - 1
        if (
            params.ma_window < 2
            or params.defense_window < 1
            or bars > 10
            or not 1 <= params.required_defenses <= params.defense_window
            or not np.isfinite(params.z_entry)
            or params.z_entry <= 0
        ):
            raise ValueError("Require valid windows totaling at most 10 bars and positive z_entry.")
        return int(bars)

    def indicators(self, data: pd.DataFrame, params: DefenseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        window = close.rolling(params.ma_window, min_periods=params.ma_window)
        out["ma"] = window.mean()
        out["scale"] = window.std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["scale"].where(out["scale"] > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        valid = (
            np.isfinite(close)
            & np.isfinite(data["high"])
            & np.isfinite(data["low"])
            & np.isfinite(out["ma"])
        )
        out["bull_defense"] = (
            (data["low"] <= out["ma"]) & (close > out["ma"])
        ).astype(float).where(valid)
        out["bear_defense"] = (
            (data["high"] >= out["ma"]) & (close < out["ma"])
        ).astype(float).where(valid)
        out["bull_count"] = out["bull_defense"].rolling(
            params.defense_window, min_periods=params.defense_window
        ).sum()
        out["bear_count"] = out["bear_defense"].rolling(
            params.defense_window, min_periods=params.defense_window
        ).sum()
        out["entry"] = np.select(
            [
                (out["z"] >= params.z_entry)
                & (out["bull_count"] >= params.required_defenses),
                (out["z"] <= -params.z_entry)
                & (out["bear_count"] >= params.required_defenses),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DefenseParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(indicators["entry"].fillna(0).to_numpy(dtype=int)):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # An opposite qualifying entry closes the position for one bar.
                position = 0
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DefenseParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0]}, index=index)
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1]
    assert (signals["size"] > 0).all()
    assert strategy.warmup_bars(params) == 9
    assert strategy.indicators(data, params)["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
