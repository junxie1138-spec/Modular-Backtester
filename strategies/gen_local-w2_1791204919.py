from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBreakParams:
    peak_window: int = 63
    change_window: int = 4
    noise_window: int = 20
    threshold: float = 1.5
    minimum_depth: float = 0.01
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownBreakParams]):
    strategy_id = "gen_local-w2_1791204919"

    @classmethod
    def params_type(cls) -> type[DrawdownBreakParams]:
        return DrawdownBreakParams

    def warmup_bars(self, params: DrawdownBreakParams) -> int:
        return int(params.peak_window + max(2 * params.change_window, params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: DrawdownBreakParams) -> pd.DataFrame:
        if min(params.peak_window, params.change_window, params.noise_window) < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive.")
        if not np.isfinite(params.minimum_depth) or not 0 <= params.minimum_depth < 1:
            raise ValueError("minimum_depth must be in [0, 1).")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be three, four, or five.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["depth_change"] = out["depth"].diff()
        out["noise"] = out["depth_change"].rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0).shift(1)
        out["recent_depth"] = out["depth"].rolling(params.change_window, min_periods=params.change_window).mean()
        out["prior_depth"] = out["recent_depth"].shift(params.change_window)
        out["contrast"] = out["prior_depth"] - out["recent_depth"]
        denominator = out["noise"].where(out["noise"] > 1e-12) * np.sqrt(params.change_window)
        out["score"] = (out["contrast"] / denominator).replace([np.inf, -np.inf], np.nan)
        out["previous_score"] = out["score"].shift(1)
        eligible = out["prior_depth"].ge(params.minimum_depth) & out["score"].notna() & out["previous_score"].notna()
        upward_break = eligible & out["score"].gt(params.threshold) & out["previous_score"].le(params.threshold)
        downward_break = eligible & out["score"].lt(-params.threshold) & out["previous_score"].ge(-params.threshold)
        out["entry"] = np.select([upward_break, downward_break], [1, -1], default=0).astype(int)
        out["strength"] = (0.25 + 0.70 * (out["score"].abs() / (2.0 * params.threshold)).clip(0.0, 1.0)).fillna(0.25)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownBreakParams) -> SignalFrame:
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be three, four, or five.")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        side, expiry, entry_size = 0, -1, 0.25
        for i in range(len(data)):
            if side and i == expiry:
                side = 0
            elif not side and entries[i]:
                side = int(entries[i])
                expiry = i + params.holding_bars
                entry_size = float(strengths[i])
            signal[i] = side
            if side:
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0], "strength": [0.8] * 5 + [0.6] * 6}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, DrawdownBreakParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].iloc[1:5].eq(0.8).all()
    assert result["size"].iloc[6:10].eq(0.6).all()
    for holding_bars in (3, 4, 5):
        indicators["entry"] = [1] + [0] * 10
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, DrawdownBreakParams(holding_bars=holding_bars)).data
        assert result["signal"].eq(1).sum() == holding_bars
        assert result["signal"].iloc[holding_bars + 1] == 0
