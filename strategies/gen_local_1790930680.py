from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    reference_window: int = 20
    change_z: float = 2.2
    hold_bars: int = 2

    def __post_init__(self):
        if not isinstance(self.reference_window, int) or self.reference_window < 3:
            raise ValueError("reference_window must be an integer >= 3")
        if not np.isfinite(self.change_z) or self.change_z <= 0:
            raise ValueError("change_z must be finite and positive")
        if not isinstance(self.hold_bars, int) or self.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = "gen_local_1790930680"

    @classmethod
    def params_type(cls):
        return ChangeParams

    def warmup_bars(self, params: ChangeParams) -> int:
        return params.reference_window + 3

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = out["return"].rolling(params.reference_window, min_periods=params.reference_window)
        # Freeze the reference distribution before both confirmation bars.
        out["reference_mean"] = window.mean().shift(2)
        out["reference_std"] = window.std(ddof=1).shift(2).replace(0.0, np.nan)
        out["first_z"] = (out["return"].shift(1) - out["reference_mean"]) / out["reference_std"]
        out["second_z"] = (out["return"] - out["reference_mean"]) / out["reference_std"]
        out["change_score"] = (out["first_z"] + out["second_z"]) / np.sqrt(2.0)
        long_confirmed = (
            (out["first_z"] > 0.5)
            & (out["second_z"] > 0.5)
            & (out["return"].shift(1) > 0)
            & (out["return"] > 0)
            & (out["change_score"] >= params.change_z)
        )
        short_confirmed = (
            (out["first_z"] < -0.5)
            & (out["second_z"] < -0.5)
            & (out["return"].shift(1) < 0)
            & (out["return"] < 0)
            & (out["change_score"] <= -params.change_z)
        )
        out["confirmed"] = np.select([long_confirmed, short_confirmed], [1, -1], default=0).astype(int)
        # Enter only on the onset of a confirmed change, not every bar of its run.
        out["entry"] = out["confirmed"].where(
            out["confirmed"].ne(out["confirmed"].shift(1).fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangeParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i - entry_bar >= params.hold_bars:
                    position = 0
            elif entry:
                position = int(entry)
                entry_bar = i
            # Opposite entries cannot close, reverse, or extend an open trade.
            # The exit bar stays flat even if another entry is present.
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    entries = pd.DataFrame({"entry": [1, -1, 1, -1, 1, 1, 0, 0]}, index=index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 0]),
        (2, [0, 1, 1, 0, -1, -1, 0, 0]),
    ):
        params = ChangeParams(reference_window=3, hold_bars=hold)
        result = strategy.generate_signals(data, entries, context, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
        assert strategy.indicators(data, params)["entry"].eq(0).all()
