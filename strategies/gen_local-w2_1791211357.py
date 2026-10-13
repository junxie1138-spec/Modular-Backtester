from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransferParams:
    gap_window: int = 20
    gap_multiple: float = 1.0
    gap_decay: float = 0.8
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapTransferParams]):
    strategy_id = "gen_local-w2_1791211357"

    @classmethod
    def params_type(cls) -> type[GapTransferParams]:
        return GapTransferParams

    @staticmethod
    def warmup_bars(params: GapTransferParams) -> int:
        return max(1, int(params.gap_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTransferParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["gap"] = (opening / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].abs().rolling(
            max(1, int(params.gap_window)), min_periods=max(1, int(params.gap_window))
        ).median().shift(1)
        out["intraday"] = (data["close"] / opening - 1).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["gap"]).fillna(0).astype(int)
        out["evidence"] = (
            (out["gap_scale"] > 0)
            & (out["gap"].abs() >= params.gap_multiple * out["gap_scale"])
            & (out["direction"] * out["intraday"] > 0)
        ).fillna(False)
        out["confirmed"] = (
            out["evidence"]
            & out["evidence"].shift(1, fill_value=False)
            & (out["direction"] == out["direction"].shift(1))
            & (out["gap"].abs() <= params.gap_decay * out["gap"].shift(1).abs())
            & (out["intraday"].abs() > out["intraday"].shift(1).abs())
        ).fillna(False)
        out["candidate"] = out["direction"].where(out["confirmed"], 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTransferParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Use a positive profit target and a one- or two-bar holding limit.")
        if params.gap_window < 1 or params.gap_multiple <= 0 or not 0 < params.gap_decay < 1:
            raise ValueError("Gap parameters must define a positive threshold and shrinking gaps.")
        decisions = np.zeros(len(data), dtype=int)
        openings = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if direction:
                # A decision on the previous close first fills at this open.
                if np.isnan(entry_price):
                    if not np.isfinite(openings[i]) or openings[i] <= 0:
                        direction = 0
                        continue
                    entry_price = openings[i]
                    entry_bar = i
                gain = direction * (closes[i] / entry_price - 1)
                held_bars = i - entry_bar + 1
                # Close-observed exits execute at the following open.
                if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.max_hold_bars:
                    direction = 0
                    entry_price = np.nan
                decisions[i] = direction
                # An exit cannot simultaneously initiate another position.
                continue
            if i >= self.warmup_bars(params) - 1 and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(candidates[i])
                entry_price = np.nan
                decisions[i] = direction
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [102.0] * 8,
        "low": [98.0] * 8,
        "close": [100.0, 100.0, 100.0, 100.3, 100.4, 100.0, 99.0, 99.0],
        "volume": [1000.0] * 8,
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapTransferParams(gap_window=1)
    evidence = pd.DataFrame({"candidate": [0, 0, 1, 0, 0, -1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, 0]
    assert (result["size"] > 0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived["candidate"].iloc[:2].eq(0).all()


if __name__ == "__main__":
    _self_check()
