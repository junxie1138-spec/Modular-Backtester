from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    lookback: int = 40
    edge: float = 0.002


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791291985"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        if not isinstance(params.lookback, int) or params.lookback < 8:
            raise ValueError("lookback must be an integer >= 8")
        if not np.isfinite(params.edge) or params.edge <= 0:
            raise ValueError("edge must be finite and positive")
        return int(10 + 4 + params.lookback)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(10, min_periods=10).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].where(out["peak"] > 0)).clip(0.0, 1.0)
        out["lag_depth"] = out["depth"].shift(4)
        paired_depth = out["depth"].where(out["lag_depth"].notna())
        out["paired_depth"] = paired_depth
        window = params.lookback
        out["mean_depth"] = paired_depth.rolling(window, min_periods=window).mean()
        out["mean_lag_depth"] = out["lag_depth"].rolling(window, min_periods=window).mean()
        out["lag_variance"] = out["lag_depth"].rolling(window, min_periods=window).var()
        out["covariance"] = paired_depth.rolling(window, min_periods=window).cov(out["lag_depth"])
        out["beta"] = out["covariance"] / out["lag_variance"].where(out["lag_variance"] > 1e-12)
        out["forecast_depth"] = (
            out["mean_depth"] + out["beta"] * (out["depth"] - out["mean_lag_depth"])
        ).clip(0.0, 1.0)
        out["forecast_edge"] = out["depth"] - out["forecast_depth"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        edge = indicators["forecast_edge"].reindex(data.index)
        candidates = np.where(
            np.isfinite(edge) & (edge > params.edge), 1,
            np.where(np.isfinite(edge) & (edge < -params.edge), -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposing entry evidence exits; neutral evidence retains the position.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame(
        {"forecast_edge": [np.nan, 0.003, 0.0, -0.003, -0.003, 0.0, 0.003, 0.003]},
        index=index,
    )
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=100),
    )
    derived = strategy.indicators(flat, params)
    assert not strategy.generate_signals(flat, derived, None, params).data["signal"].any()
    assert strategy.warmup_bars(params) >= 10 + 4 + params.lookback - 1


if __name__ == "__main__":
    _self_check()
