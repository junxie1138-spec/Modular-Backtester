from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    participation_threshold: float = 0.65
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791321269"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.5 < params.participation_threshold < 1.0:
            raise ValueError("participation_threshold must be between 0.5 and 1")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["up_volume"] = data["volume"].where(
            out["change"].notna()
        ).where(out["change"] > 0, 0.0).where(out["change"].notna())
        out["volume_3"] = data["volume"].rolling(3, min_periods=3).sum()
        out["volume_12"] = data["volume"].rolling(12, min_periods=12).sum()
        out["up_volume_3"] = out["up_volume"].rolling(3, min_periods=3).sum()
        out["up_volume_12"] = out["up_volume"].rolling(12, min_periods=12).sum()
        out["participation"] = out["up_volume_3"] / out["volume_3"].where(out["volume_3"] > 0)
        out["background"] = (
            out["up_volume_12"] / out["volume_12"].where(out["volume_12"] > 0)
        ).shift(1)
        out["previous_participation"] = out["participation"].shift(1)
        out["volume_baseline"] = data["volume"].shift(1).rolling(20, min_periods=20).mean()
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        out["entry"] = (
            (out["participation"] > params.participation_threshold)
            & (out["previous_participation"] <= params.participation_threshold)
            & out["background"].between(0.30, 0.55)
            & (out["volume_ratio"] > 1.10)
            & (out["change"] > 0.35 * out["atr"])
            & (data["close"] > data["open"])
            & (out["close_location"] >= 0.70)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_atr = 0.0
        stop_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Fill uses this bar's open; ATR was frozen at the decision close.
                    stop_price = opens[i] - params.stop_atr * entry_atr
                if (
                    not np.isfinite(close[i])
                    or not np.isfinite(stop_price)
                    or close[i] < stop_price
                    or i - entry_bar + 1 >= 2
                ):
                    entry_bar = -1
                    continue
                desired[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                entry_atr = atr[i]
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [101.0] * 6,
        "low": [97.0] * 6,
        "close": [100.0, 99.0, 100.0, 100.0, 98.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [True, False, False, True, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 0]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, Params())["entry"].any()


if __name__ == "__main__":
    _self_check()
