from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_window: int = 5
    baseline_window: int = 30
    min_disparity: float = 0.35
    hold_bars: int = 4
    target_vol: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791284933"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.short_window, params.baseline_window) + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.short_window < 2 or params.baseline_window < 2:
            raise ValueError("Volatility windows must be at least two bars")
        if params.hold_bars < 3 or params.hold_bars > 5:
            raise ValueError("hold_bars must be between three and five")
        if not np.isfinite([params.min_disparity, params.target_vol, params.max_size]).all():
            raise ValueError("Sizing and disparity parameters must be finite")
        if params.min_disparity <= 0 or params.target_vol <= 0 or not 0 < params.max_size <= 1:
            raise ValueError("Invalid disparity or exposure parameters")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous_close = close.shift(1)
        returns = close.div(previous_close).sub(1)
        true_range = pd.concat([
            data["high"].sub(data["low"]),
            data["high"].sub(previous_close).abs(),
            data["low"].sub(previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["range_vol"] = true_range.div(previous_close).rolling(params.short_window).mean().div(1.6)
        out["return_vol"] = returns.rolling(params.short_window).std(ddof=0)
        out["baseline_vol"] = returns.rolling(params.baseline_window).std(ddof=0)
        denominator = out["return_vol"].where(out["return_vol"] > 0)
        out["disparity"] = out["range_vol"].div(denominator).sub(1)
        out["convergence"] = out["disparity"].shift(1).sub(out["disparity"])
        out["range_change"] = out["range_vol"].diff()
        out["return_change"] = out["return_vol"].diff()
        bar_range = data["high"].sub(data["low"])
        out["close_location"] = close.sub(data["low"]).div(bar_range.where(bar_range > 0))
        out["strength"] = out["convergence"].div(params.min_disparity).clip(0.20, 1.0)
        annual_vol = out[["return_vol", "baseline_vol"]].max(axis=1) * np.sqrt(252.0)
        out["proposed_size"] = (
            params.target_vol / annual_vol.where(annual_vol > 0) * out["strength"]
        ).clip(0.01, params.max_size)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["range_vol", "return_vol", "baseline_vol", "disparity", "convergence", "proposed_size"]].notna().all(axis=1)
        out["entry_ready"] = (
            ready
            & out["disparity"].shift(1).ge(params.min_disparity)
            & out["disparity"].ge(0)
            & out["convergence"].ge(params.min_disparity * 0.20)
            & out["range_change"].lt(0)
            & out["return_change"].gt(0)
            & out["close_location"].ge(0.65)
            & returns.gt(0)
        ).fillna(False)
        out["proposed_size"] = out["proposed_size"].fillna(0.01)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        eligible = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        proposed = indicators["proposed_size"].to_numpy(dtype=float)
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if i < exit_bar:
                signals[i] = 1
                sizes[i] = entry_size
            elif i == exit_bar:
                continue
            elif eligible[i] and np.isfinite(proposed[i]) and proposed[i] > 0:
                entry_size = float(proposed[i])
                exit_bar = i + params.hold_bars
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "entry_ready": [True, False, True, False, True, False, False, False],
        "proposed_size": [0.4, 0.2, 0.8, 0.2, 0.7, 0.2, 0.2, 0.2],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.allclose(result["size"].iloc[1:5], 0.4)
