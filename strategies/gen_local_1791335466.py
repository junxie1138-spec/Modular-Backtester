from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    trend_window: int = 8
    drawdown_window: int = 40


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791335466"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.trend_window + 3,
                       params.drawdown_window + params.trend_window - 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        n = params.trend_window
        d = params.drawdown_window
        if n < 2 or d < 2:
            raise ValueError("Both windows must be at least two bars")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["time"] = np.arange(len(data), dtype=float)
        out["mean_log_close"] = out["log_close"].rolling(n).mean()
        out["mean_time"] = out["time"].rolling(n).mean()
        out["mean_time_price"] = (
            out["time"] * out["log_close"]
        ).rolling(n).mean()
        out["latent_slope"] = (
            out["mean_time_price"]
            - out["mean_time"] * out["mean_log_close"]
        ) / ((n * n - 1) / 12.0)
        out["fitted_close"] = (
            out["mean_log_close"]
            + out["latent_slope"] * ((n - 1) / 2.0)
        )
        out["forecast"] = (
            out["fitted_close"] + out["latent_slope"]
        ).shift(1)
        out["innovation"] = out["log_close"] - out["forecast"]
        out["innovation_pressure"] = out["innovation"].rolling(3).mean()
        out["peak"] = close.rolling(d).max()
        out["drawdown"] = close / out["peak"].where(out["peak"] > 0) - 1.0
        out["recent_drawdown"] = out["drawdown"].rolling(n).min()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        recovery = (
            indicators["recent_drawdown"].le(-0.015)
            & indicators["latent_slope"].gt(0.0)
            & indicators["innovation_pressure"].gt(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; its reversal exits.
        df["signal"] = recovery.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RecoveryParams()
    prices = np.concatenate([
        np.full(60, 100.0),
        np.linspace(100.0, 92.0, 20),
        np.linspace(92.0, 105.0, 20),
        np.full(20, 105.0),
    ])
    data = pd.DataFrame(
        {"open": prices, "high": prices + 0.5,
         "low": prices - 0.5, "close": prices, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(prices), freq="B"),
    )
    ind = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, ind, None, params).data
    expected = (
        ind["recent_drawdown"].le(-0.015)
        & ind["latent_slope"].gt(0.0)
        & ind["innovation_pressure"].gt(0.0)
    ).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(signals["signal"], expected, check_names=False)
    assert signals["signal"].isin([0, 1]).all()
    assert signals["signal"].max() == 1
    assert signals["signal"].iloc[-1] == 0
    assert signals["size"].gt(0).all()
    prefix = GeneratedStrategy.indicators(data.iloc[:90], params)
    pd.testing.assert_frame_equal(prefix, ind.iloc[:90])
