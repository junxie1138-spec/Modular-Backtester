from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    depth_window: int = 63
    echo_lag: int = 18
    correlation_window: int = 63
    threshold: float = 0.2


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791327693"

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    @staticmethod
    def warmup_bars(params: EchoParams) -> int:
        if min(params.depth_window, params.echo_lag, params.correlation_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("Threshold must be finite and positive.")
        return int(params.depth_window + params.echo_lag + params.correlation_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        out["peak"] = out["log_close"].rolling(params.depth_window, min_periods=params.depth_window).max()
        out["trough"] = out["log_close"].rolling(params.depth_window, min_periods=params.depth_window).min()
        out["drawdown"] = out["peak"] - out["log_close"]
        out["drawup"] = out["log_close"] - out["trough"]
        # Log-price reflection swaps drawdown and drawup, reversing pressure.
        out["pressure"] = out["drawdown"] - out["drawup"]
        out["shock"] = out["pressure"].diff()
        out["lagged_shock"] = out["shock"].shift(params.echo_lag)
        out["echo"] = out["shock"].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out["lagged_shock"]).clip(-1.0, 1.0)
        out["recent_shock"] = out["shock"].rolling(params.echo_lag, min_periods=params.echo_lag).mean()
        out["shock_scale"] = out["shock"].rolling(params.correlation_window, min_periods=params.correlation_window).std(ddof=0)
        scale = out["shock_scale"].where(out["shock_scale"] > 1e-12)
        out["score"] = (out["echo"] * out["recent_shock"] / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        score = indicators["score"].reindex(data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        events = np.where(score <= -params.threshold, 1, np.where(score >= params.threshold, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, event in enumerate(events):
            if position == 0:
                position = int(event)
            elif event == -position:
                # The opposite entry condition closes either side symmetrically.
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = EchoParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [0.0, -0.3, 0.0, 0.3, 0.3, 0.0, -0.3, -0.3]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    mirrored = GeneratedStrategy.generate_signals(data, -indicators, None, params).data
    assert mirrored["signal"].tolist() == (-result["signal"]).tolist()
    assert (result["size"] > 0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert flat["score"].eq(0.0).all()


if __name__ == "__main__":
    _self_check()
