from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetestParams:
    gap_threshold: float = 0.35
    profit_target: float = 0.008


class GeneratedStrategy(BaseStrategy[GapRetestParams]):
    strategy_id = "gen_local-w2_1791286269"

    @classmethod
    def params_type(cls) -> type[GapRetestParams]:
        return GapRetestParams

    @staticmethod
    def warmup_bars(params: GapRetestParams) -> int:
        return 11

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRetestParams) -> pd.DataFrame:
        if not np.isfinite(params.gap_threshold) or params.gap_threshold <= 0:
            raise ValueError("gap_threshold must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")

        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].rolling(5, min_periods=5).max().shift(1)
        out["prior_low"] = data["low"].rolling(5, min_periods=5).min().shift(1)
        out["range_scale"] = (
            (data["high"] - data["low"])
            .rolling(10, min_periods=10).median().shift(1)
        )
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_score"] = out["gap"] / out["range_scale"].where(out["range_scale"] > 0)

        valid = (
            np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & (data[["open", "high", "low", "close"]] > 0).all(axis=1)
            & np.isfinite(out).all(axis=1)
        )
        long_entry = (
            valid
            & (out["gap_score"] >= params.gap_threshold)
            & (data["open"] > out["prior_high"])
            & (data["low"] <= out["prior_high"])
            & (data["close"] > data["open"])
        )
        short_entry = (
            valid
            & (out["gap_score"] <= -params.gap_threshold)
            & (data["open"] < out["prior_low"])
            & (data["high"] >= out["prior_low"])
            & (data["close"] < data["open"])
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRetestParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")

        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        held_bars = 0
        entry_price = np.nan

        for i in range(len(data)):
            if position != 0:
                if held_bars == 0:
                    # The preceding close's decision fills at this bar's open.
                    entry_price = opens[i]
                held_bars += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = (
                    valid_entry
                    and np.isfinite(closes[i])
                    and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                )
                # Targets are observed at the close; exits fill next open.
                if target_hit or held_bars >= 2 or not valid_entry:
                    position = 0
                    held_bars = 0
                    entry_price = np.nan
            elif entries[i] != 0:
                position = int(entries[i])
                held_bars = 0
                entry_price = np.nan
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [103.0] * 6,
        "low": [98.0] * 6,
        "close": [100.0, 101.0, 102.0, 100.0, 99.0, 99.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0]}, index=index)
    target = GeneratedStrategy.generate_signals(
        data, indicators, None, GapRetestParams(profit_target=0.005)
    ).data
    timed = GeneratedStrategy.generate_signals(
        data, indicators, None, GapRetestParams(profit_target=0.5)
    ).data
    assert target["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    assert timed["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (target["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(GapRetestParams()) >= 11


if __name__ == "__main__":
    _self_check()
