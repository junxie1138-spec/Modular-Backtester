from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDrawdownParams:
    peak_window: int = 8
    gap_window: int = 3
    min_drawdown: float = 0.012
    min_depth_change: float = 0.003
    min_gap_coherence: float = 0.65
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapDrawdownParams]):
    strategy_id = "gen_local-w2_1791287296"

    @classmethod
    def params_type(cls) -> type[GapDrawdownParams]:
        return GapDrawdownParams

    @staticmethod
    def warmup_bars(params: GapDrawdownParams) -> int:
        if not 1 <= params.gap_window < params.peak_window <= 10:
            raise ValueError("Require 1 <= gap_window < peak_window <= 10")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        if not np.isfinite(params.min_drawdown) or not 0 < params.min_drawdown < 1:
            raise ValueError("min_drawdown must be finite and between zero and one")
        if not np.isfinite(params.min_depth_change) or not 0 < params.min_depth_change < 1:
            raise ValueError("min_depth_change must be finite and between zero and one")
        if not np.isfinite(params.min_gap_coherence) or not 0 < params.min_gap_coherence <= 1:
            raise ValueError("min_gap_coherence must be finite and in (0, 1]")
        return int(max(params.peak_window, params.gap_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapDrawdownParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        gap = opening.div(close.shift(1)).sub(1)
        gap_net = gap.rolling(params.gap_window, min_periods=params.gap_window).sum()
        gap_total = gap.abs().rolling(params.gap_window, min_periods=params.gap_window).sum()
        depth = peak.sub(close).div(peak)
        prior_depth = peak.sub(close.shift(params.gap_window)).div(peak)
        result = pd.DataFrame({
            "peak": peak,
            "gap": gap,
            "gap_net": gap_net,
            "gap_total": gap_total,
            "gap_coherence": gap_net.abs().div(gap_total.replace(0, np.nan)),
            "depth": depth,
            "prior_depth": prior_depth,
            "depth_change": depth.sub(prior_depth),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDrawdownParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        columns = ["depth", "depth_change", "gap_net", "gap_coherence"]
        valid = indicators[columns].notna().all(axis=1)
        eligible = valid & (indicators["depth"] >= params.min_drawdown)
        eligible &= indicators["gap_coherence"] >= params.min_gap_coherence
        buy = eligible & (indicators["gap_net"] < 0)
        buy &= indicators["depth_change"] <= -params.min_depth_change
        sell = eligible & (indicators["gap_net"] > 0)
        sell &= indicators["depth_change"] >= params.min_depth_change
        candidates = np.select([buy, sell], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar == params.hold_bars:
                    position = 0
                signals[i] = position
                continue
            if candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapDrawdownParams()
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"depth": 0.02, "depth_change": 0.0,
                             "gap_net": 0.0, "gap_coherence": 1.0}, index=index)
    evidence.loc[index[0], ["depth_change", "gap_net"]] = [-0.004, -0.01]
    evidence.loc[index[1], ["depth_change", "gap_net"]] = [0.004, 0.01]
    evidence.loc[index[9], ["depth_change", "gap_net"]] = [0.004, 0.01]
    actual = GeneratedStrategy.generate_signals(data, evidence, None, params).data
    expected = np.zeros(20, dtype=int)
    expected[1:9] = 1
    expected[10:18] = -1
    assert np.array_equal(actual["signal"].to_numpy(), expected)
    assert (actual["size"] > 0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.warmup_bars(params) <= 10
    assert not GeneratedStrategy.generate_signals(data, flat, None, params).data["signal"].any()
