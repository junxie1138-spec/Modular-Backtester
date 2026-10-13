from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    short_window: int = 5
    baseline_window: int = 40
    compression_ratio: float = 0.75
    min_gap: float = 0.0015
    rejection_fraction: float = 0.8
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local-w2_1791333163"

    @classmethod
    def params_type(cls) -> type[GapRejectionParams]:
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        if not 2 <= params.short_window <= params.baseline_window:
            raise ValueError("Require 2 <= short_window <= baseline_window")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not 0 < params.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if not 0 < params.rejection_fraction <= 1:
            raise ValueError("rejection_fraction must be in (0, 1]")
        if not np.isfinite(params.min_gap) or params.min_gap <= 0:
            raise ValueError("min_gap must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        return int(params.baseline_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda x: x > 0)
        out["range_pct"] = (data["high"] - data["low"]) / out["previous_close"]
        out["short_range"] = out["range_pct"].rolling(params.short_window).mean().shift(1)
        out["baseline_range"] = out["range_pct"].rolling(params.baseline_window).mean().shift(1)
        out["channel_high"] = data["high"].rolling(params.short_window).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.short_window).min().shift(1)
        out["gap"] = data["open"] / out["previous_close"] - 1.0
        out["intraday_move"] = (data["close"] - data["open"]) / out["previous_close"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["gap_direction"] = np.sign(out["gap"]).fillna(0).astype(int)
        out["compressed"] = (
            (out["baseline_range"] > 0)
            & (out["short_range"] <= params.compression_ratio * out["baseline_range"])
        )
        out["rejected"] = (
            out["compressed"]
            & (out["gap"].abs() >= params.min_gap)
            & (out["gap_direction"] * out["intraday_move"] <= -params.rejection_fraction * out["gap"].abs())
            & (data["close"] >= out["channel_low"])
            & (data["close"] <= out["channel_high"])
            & (data["open"] > 0)
            & (data["close"] > 0)
        ).fillna(False)
        out["candidate"] = (-out["gap_direction"]).where(out["rejected"], 0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].ne(0) & out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRejectionParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Capture the entry open only when its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                if position:
                    gain = position * (closes[i] / entry_price - 1.0)
                    # Targets are evaluated at the close; exits fill next open.
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(confirmed[i])
                entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapRejectionParams()
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({
        "open": [100.0] * 12,
        "high": [103.0] * 12,
        "low": [97.0] * 12,
        "close": [100.0, 100.0, 102.0] + [100.0] * 9,
        "volume": [1000.0] * 12,
    }, index=index)
    indicators = pd.DataFrame({"confirmed": [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0]
    data.loc[index[2], "close"] = 98.0
    indicators["confirmed"] = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist()[:4] == [0, -1, -1, 0]
