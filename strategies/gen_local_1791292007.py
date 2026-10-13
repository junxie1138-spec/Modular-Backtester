from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationParams:
    entry_innovation: float = 0.2
    regime_threshold: float = 0.25
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[InnovationParams]):
    strategy_id = "gen_local_1791292007"

    @classmethod
    def params_type(cls) -> type[InnovationParams]:
        return InnovationParams

    def warmup_bars(self, params: InnovationParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: InnovationParams) -> pd.DataFrame:
        log_price = np.log(data["close"].where(data["close"] > 0))
        returns = log_price.diff()
        # Exact least-squares slope of three equally spaced log closes.
        slope = (log_price - log_price.shift(2)) / 2.0
        innovation = slope.diff()
        scale = returns.rolling(6, min_periods=6).std(ddof=0).replace(0.0, np.nan)
        correlation = innovation.rolling(4, min_periods=4).corr(innovation.shift(1))
        result = pd.DataFrame({
            "log_price": log_price,
            "returns": returns,
            "latent_slope": slope,
            "slope_innovation": innovation,
            "return_scale": scale,
            "normalized_slope": slope / scale,
            "normalized_innovation": innovation / scale,
            "innovation_correlation": correlation,
        }, index=data.index)
        result = result.replace([np.inf, -np.inf], np.nan)
        result["ready"] = result[["normalized_slope", "normalized_innovation", "innovation_correlation"]].notna().all(axis=1)
        return result.fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: InnovationParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if not np.isfinite(params.entry_innovation) or params.entry_innovation <= 0:
            raise ValueError("entry_innovation must be finite and positive")
        if not np.isfinite(params.regime_threshold) or not 0 < params.regime_threshold < 1:
            raise ValueError("regime_threshold must be between zero and one")

        slope = indicators["normalized_slope"].to_numpy(dtype=float)
        innovation = indicators["normalized_innovation"].to_numpy(dtype=float)
        correlation = indicators["innovation_correlation"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        regime = 0
        exit_bar = -1
        first_decision = self.warmup_bars(params) - 1

        for i in range(len(data)):
            valid = ready[i] and np.isfinite(slope[i]) and np.isfinite(innovation[i]) and np.isfinite(correlation[i])
            if valid and i >= first_decision:
                if correlation[i] >= params.regime_threshold:
                    regime = 1
                elif correlation[i] <= -params.regime_threshold:
                    regime = -1

            # Positions survive missing indicators and all regime changes.
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue

            if not valid or i < first_decision:
                continue
            launch = regime == 1 and slope[i] > 0.0
            rebound = regime == -1 and slope[i] < 0.0
            if (launch or rebound) and innovation[i] >= params.entry_innovation:
                signals[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = InnovationParams()
    strategy = GeneratedStrategy()
    warmup = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=warmup + params.hold_bars + 1)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    forced = pd.DataFrame({"normalized_slope": 1.0,
                           "normalized_innovation": 1.0,
                           "innovation_correlation": 0.8,
                           "ready": True}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[warmup:warmup + params.hold_bars] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert not flat.isna().any().any()
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
