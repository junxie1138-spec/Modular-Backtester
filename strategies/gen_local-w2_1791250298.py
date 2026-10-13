from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 20
    percentile_window: int = 96
    innovation_percentile: float = 0.85
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791250298"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.trend_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.trend_window < 3 or params.percentile_window < 3:
            raise ValueError("Lookback windows must be at least three bars")
        if not 0.5 < params.innovation_percentile < 1.0:
            raise ValueError("Innovation percentile must be between 0.5 and 1")
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Exit parameters must be positive")

        out = pd.DataFrame(index=data.index)
        length = params.trend_window
        center = (length - 1) / 2.0
        denominator = length * (length * length - 1) / 12.0
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_range"] = np.log(
            data["high"].where(data["high"] > 0)
            / data["low"].where(data["low"] > 0)
        ).clip(lower=0)
        out["range_scale"] = out["log_range"].rolling(length).mean().replace(0, np.nan)
        out["trend_mean"] = out["log_close"].rolling(length).mean()
        weighted = pd.Series(0.0, index=data.index)
        for lag in range(length):
            weighted = weighted + (center - lag) * out["log_close"].shift(lag)
        out["slope"] = weighted / denominator
        out["trend_endpoint"] = out["trend_mean"] + center * out["slope"]
        out["forecast"] = out["trend_endpoint"].shift(1) + out["slope"].shift(1)
        out["residual"] = out["log_close"] - out["forecast"]
        out["slope_innovation"] = out["slope"].diff() / out["range_scale"]
        out["innovation_threshold"] = out["slope_innovation"].rolling(
            params.percentile_window
        ).quantile(params.innovation_percentile).shift(1)
        out["residual_median"] = out["residual"].rolling(
            params.percentile_window
        ).quantile(0.5).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        residual = indicators["residual"]
        median = indicators["residual_median"]
        entry = (
            (residual > median)
            & (residual.shift(1) <= median.shift(1))
            & (indicators["slope"] > 0)
            & (indicators["slope_innovation"] > 0)
            & (indicators["slope_innovation"] > indicators["innovation_threshold"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held = 0

        for i in range(len(data)):
            if active:
                if held == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                held += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or held >= params.max_hold_bars:
                    active = False
                    held = 0
                else:
                    desired[i] = 1
            elif entry[i]:
                active = True
                held = 0
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0,
         "close": [100.0, 100.0, 103.0, 100.0, 100.0, 100.0], "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"residual": [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         "residual_median": 0.0, "slope": 1.0,
         "slope_innovation": 2.0, "innovation_threshold": 1.0},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(
        data, indicators, None, StrategyParams(max_hold_bars=2)
    ).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
