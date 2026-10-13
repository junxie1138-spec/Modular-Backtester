from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    forecast_threshold: float = 0.15
    correlation_floor: float = 0.20
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        values = (self.forecast_threshold, self.correlation_floor, self.profit_target)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Thresholds must be finite.")
        if self.forecast_threshold <= 0 or not 0 <= self.correlation_floor < 1:
            raise ValueError("Invalid innovation thresholds.")
        if self.profit_target <= 0 or self.max_hold_bars < 1:
            raise ValueError("Profit target and holding limit must be positive.")


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local-w2_1791308327"

    @classmethod
    def params_type(cls):
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        # Three-close least-squares slope is a finite-memory latent trend estimate.
        out["slope"] = out["log_close"].diff(2) / 2.0
        out["innovation"] = out["slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["innovation_scale"] = out["innovation"].rolling(5, min_periods=5).std(ddof=0)
        out["innovation_correlation"] = out["innovation"].rolling(5, min_periods=5).corr(out["lagged_innovation"])
        out["forecast"] = (
            out["innovation_correlation"] * out["innovation"]
            / out["innovation_scale"].where(out["innovation_scale"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["entry_ready"] = (
            (out["slope"] > 0)
            & (out["innovation"] > 0)
            & (out["innovation_correlation"] > params.correlation_floor)
            & (out["forecast"] > params.forecast_threshold)
            & (out["forecast"].shift(1).fillna(0) <= params.forecast_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: InnovationParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                # Record today's open only after yesterday's entry decision.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        entry_price = np.nan
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    active = False
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_price = np.nan
                desired[i] = 1
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 104.0
    indicators = pd.DataFrame({"entry_ready": False}, index=index)
    indicators.loc[index[[0, 4]], "entry_ready"] = True
    strategy = GeneratedStrategy()
    params = InnovationParams(max_hold_bars=5)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    actual = strategy.indicators(data, params)
    assert not actual["entry_ready"].iloc[:8].any()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
