from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecruitmentParams:
    window: int = 4
    recruitment_ratio: float = 1.25
    acceptance_min: float = 0.55


class GeneratedStrategy(BaseStrategy[RecruitmentParams]):
    strategy_id = "gen_local_1790963120"

    @classmethod
    def params_type(cls):
        return RecruitmentParams

    def warmup_bars(self, params: RecruitmentParams) -> int:
        return int(params.window) + 2

    def indicators(self, data: pd.DataFrame, params: RecruitmentParams) -> pd.DataFrame:
        if isinstance(params.window, bool) or not isinstance(params.window, int) or params.window < 2:
            raise ValueError("window must be an integer >= 2")
        if not np.isfinite(params.recruitment_ratio) or params.recruitment_ratio <= 0:
            raise ValueError("recruitment_ratio must be finite and positive")
        if not np.isfinite(params.acceptance_min) or not 0 <= params.acceptance_min <= 1:
            raise ValueError("acceptance_min must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        valid = out["return"].notna() & out["previous_return"].notna()
        out["recruited_gain"] = out["return"].clip(lower=0).where(out["previous_return"] <= 0, 0.0).where(valid)
        out["defection_loss"] = (-out["return"]).clip(lower=0).where(out["previous_return"] > 0, 0.0).where(valid)
        w = params.window
        out["gain_pool"] = out["recruited_gain"].rolling(w, min_periods=w).sum()
        out["loss_pool"] = out["defection_loss"].rolling(w, min_periods=w).sum()
        out["net_return"] = out["return"].rolling(w, min_periods=w).sum()
        out["range_width"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["accepted_fraction"] = ((data["high"] - close.shift(1)) / out["range_width"]).clip(0.0, 1.0).replace([np.inf, -np.inf], np.nan)
        out["acceptance"] = out["accepted_fraction"].rolling(w, min_periods=w).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecruitmentParams,
    ) -> SignalFrame:
        recruitment = (
            (indicators["gain_pool"] > params.recruitment_ratio * indicators["loss_pool"])
            & (indicators["net_return"] > 0)
        )
        acceptance = indicators["acceptance"] >= params.acceptance_min
        # Holding continues only while both entry primitives agree.
        df = pd.DataFrame(index=data.index)
        df["signal"] = (recruitment & acceptance).fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = pd.Series([100, 99, 100, 101, 102, 103, 102, 101, 100, 101, 102, 103], dtype=float)
    sample = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0})
    sample.index = pd.date_range("2020-01-01", periods=len(sample))
    strategy = GeneratedStrategy()
    params = RecruitmentParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    derived = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, derived, ctx, params).data
    assert result["signal"].iloc[5] == 0 and result["signal"].iloc[6] == 1
    assert result["signal"].iloc[7] == 0
    for end in range(1, len(sample) + 1):
        prefix = sample.iloc[:end]
        prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        pd.testing.assert_frame_equal(prefix_result, result.iloc[:end])
    assert set(result["signal"].unique()) <= {0, 1}
    assert (result["size"] > 0).all()
