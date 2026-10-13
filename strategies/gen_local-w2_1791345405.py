from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    innovation_threshold: float = 1.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791345405"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive and finite")
        if not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        out["time"] = np.arange(len(data), dtype=float)
        out["time_variance"] = out["time"].rolling(20).var()
        out["trend_mean"] = out["log_close"].rolling(20).mean()
        out["slope"] = out["log_close"].rolling(20).cov(out["time"]) / out["time_variance"]
        out["trend_endpoint"] = out["trend_mean"] + 9.5 * out["slope"]
        out["forecast"] = out["trend_endpoint"].shift(1) + out["slope"].shift(1)
        out["innovation"] = out["log_close"] - out["forecast"]
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(20).std(ddof=0).replace(0, np.nan)
        out["innovation_z"] = (out["innovation"] / out["volatility"]).replace([np.inf, -np.inf], np.nan)
        out["peak"] = close.rolling(63).max()
        out["drawdown"] = close / out["peak"] - 1.0
        out["slope_innovation"] = out["slope"].diff()
        out["entry_ready"] = (
            (out["drawdown"] <= -0.03)
            & (out["slope"].shift(1) < 0)
            & (out["slope_innovation"] > 0)
            & (out["innovation_z"] >= params.innovation_threshold)
            & (out["innovation"].shift(1) > 0)
            & (out["time"] >= self.warmup_bars(params) - 1)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if ready[i]:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams(holding_bars=18)
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(index=index)
    ready = pd.DataFrame({"entry_ready": True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ready, None, params).data
    expected = np.zeros(45, dtype=int)
    expected[1:19] = 1
    expected[20:38] = 1
    expected[39:] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
