from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSpringParams:
    lookback: int = 90
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapSpringParams]):
    strategy_id = "gen_local_1790968932"

    @classmethod
    def params_type(cls) -> type[GapSpringParams]:
        return GapSpringParams

    def warmup_bars(self, params: GapSpringParams) -> int:
        if params.lookback < 20 or params.hold_bars < 1:
            raise ValueError("lookback must be >= 20 and hold_bars must be >= 1")
        return int(params.lookback + 10)

    def indicators(self, data: pd.DataFrame, params: GapSpringParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        out["gap"] = np.log(opening / previous_close).replace([np.inf, -np.inf], np.nan)
        out["pressure"] = out["gap"].rolling(5, min_periods=5).sum()
        out["lagged_pressure"] = out["pressure"].shift(5)
        window = params.lookback
        current = out["pressure"].rolling(window, min_periods=window)
        lagged = out["lagged_pressure"].rolling(window, min_periods=window)
        out["mean"] = current.mean()
        out["lagged_mean"] = lagged.mean()
        out["std"] = current.std()
        out["lagged_variance"] = lagged.var()
        out["covariance"] = current.cov(out["lagged_pressure"])
        out["correlation"] = current.corr(out["lagged_pressure"])
        out["beta"] = out["covariance"] / out["lagged_variance"].where(out["lagged_variance"] > 1e-16)
        out["forecast"] = out["mean"] + out["beta"] * (out["pressure"] - out["lagged_mean"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["correlation"] < -0.15)
            & (out["pressure"] < 0.0)
            & (out["pressure"] < out["mean"] - out["std"])
            & (out["forecast"] > 0.5 * out["std"])
            & (out["std"] > 1e-8)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSpringParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    positions[i] = 1
            elif entries[i]:
                positions[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapSpringParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    assert not strategy.indicators(data, params)["entry"].any()
    entries = pd.DataFrame({"entry": True}, index=index)
    for holding in (5, 8, 10):
        params.hold_bars = holding
        result = strategy.generate_signals(data, entries, None, params).data
        assert result["signal"].iloc[0] == 0
        assert result["signal"].iloc[1:holding + 1].eq(1).all()
        assert result["signal"].iloc[holding + 1] == 0
        assert result["signal"].iloc[holding + 2] == 1
        assert result["signal"].isin([0, 1]).all()
        assert result["size"].gt(0).all()
