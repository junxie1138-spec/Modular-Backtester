from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapClearanceParams:
    gap_window: int = 60
    gap_quantile: float = 0.85
    breakout_window: int = 10
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapClearanceParams]):
    strategy_id = "gen_local_1791248294"

    @classmethod
    def params_type(cls):
        return GapClearanceParams

    def warmup_bars(self, params: GapClearanceParams) -> int:
        if params.gap_window < 2 or params.breakout_window < 1:
            raise ValueError("Lookbacks must be positive; gap_window must be at least two.")
        if not 0.0 < params.gap_quantile < 1.0:
            raise ValueError("gap_quantile must lie strictly between zero and one.")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two.")
        return int(max(params.gap_window + 2, params.breakout_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapClearanceParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        denominator = out["prior_close"].where(out["prior_close"] > 0)
        out["gap"] = (data["open"] / denominator - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_threshold"] = out["gap"].abs().shift(1).rolling(
            params.gap_window, min_periods=params.gap_window
        ).quantile(params.gap_quantile)
        out["prior_ceiling"] = data["high"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapClearanceParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entry = (
            (indicators["gap"] > indicators["gap_threshold"])
            & (indicators["gap"] > 0.0)
            & (data["low"] > indicators["prior_ceiling"])
            & (data["close"] > data["open"])
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, eligible in enumerate(entry):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                continue
            elif eligible:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame(
        {"gap": 0.02, "gap_threshold": 0.01, "prior_ceiling": 99.0}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for holding, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1]),
    ):
        params = GapClearanceParams(holding_bars=holding)
        result = strategy.generate_signals(data, features, ctx, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    params = GapClearanceParams()
    actual = strategy.indicators(data, params)
    assert actual["gap_threshold"].isna().all()
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
