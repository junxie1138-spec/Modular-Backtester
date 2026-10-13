from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_bars: int = 24
    percentile_bars: int = 252
    hub_percentile: float = 0.8
    seasonal_bars: int = 504
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791303214"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.graph_bars + params.percentile_bars + 1,
                   params.seasonal_bars + params.hold_bars + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.graph_bars < 2 or params.percentile_bars < 2
                or params.seasonal_bars < 60 or params.hold_bars < 1
                or not 0.0 < params.hub_percentile < 1.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace(
            [np.inf, -np.inf], np.nan)
        out["phase"] = np.minimum((data.index.day - 1) // 10, 2)
        price = out["log_close"]
        upper = np.zeros(len(data), dtype=float)
        lower = np.zeros(len(data), dtype=float)
        max_slope = np.full(len(data), -np.inf)
        min_slope = np.full(len(data), np.inf)
        # A natural-visibility edge exists when its slope clears all nearer slopes.
        for lag in range(1, params.graph_bars + 1):
            slope = ((price.shift(lag) - price) / lag).to_numpy()
            upper += np.isfinite(slope) & (slope > max_slope)
            lower += np.isfinite(slope) & (slope < min_slope)
            max_slope = np.fmax(max_slope, slope)
            min_slope = np.fmin(min_slope, slope)
        complete = price.rolling(params.graph_bars + 1).count().eq(params.graph_bars + 1)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(complete)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(complete)
        # The adjacent edge belongs to both graphs and is counted once.
        out["hub_degree"] = out["upper_degree"] + out["lower_degree"] - 1.0
        out["hub_threshold"] = out["hub_degree"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.hub_percentile)
        # Train only on holding-period outcomes that ended before this close.
        out["matured_return"] = price.diff(params.hold_bars).shift(1)
        out["origin_phase"] = out["phase"].shift(params.hold_bars + 1)
        seasonal = pd.Series(np.nan, index=data.index)
        for phase in range(3):
            sample = out["origin_phase"].eq(phase) & out["matured_return"].notna()
            count = sample.astype(float).rolling(
                params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            total = out["matured_return"].where(sample, 0.0).rolling(
                params.seasonal_bars, min_periods=params.seasonal_bars).sum()
            out[f"phase_count_{phase}"] = count
            out[f"phase_mean_{phase}"] = (total / count.replace(0.0, np.nan)).where(count >= 20)
            seasonal = seasonal.where(
                out["phase"].ne(phase), out[f"phase_mean_{phase}"])
        out["seasonal_return"] = seasonal
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = (indicators["hub_degree"].notna()
                 & indicators["hub_threshold"].notna()
                 & indicators["seasonal_return"].notna())
        trigger = ready & indicators["hub_degree"].gt(indicators["hub_threshold"])
        candidates = np.sign(indicators["seasonal_return"].fillna(0.0)).astype(int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar in range(self.warmup_bars(params), len(data)):
            if position != 0 and bar == exit_bar:
                position = 0
                # Keep this bar flat; a fresh entry may start on the next bar.
            elif position == 0 and bool(trigger.iloc[bar]):
                position = int(candidates.iloc[bar])
                if position != 0:
                    exit_bar = bar + params.hold_bars
            decisions[bar] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    length = strategy.warmup_bars(params) + 2 * params.hold_bars + 6
    index = pd.bdate_range("2020-01-01", periods=length)
    close = 100.0 * np.exp(np.arange(length) * 0.0002)
    data = pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1000000.0}, index=index)
    derived = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-7], params)
    pd.testing.assert_frame_equal(derived.iloc[:-7], prefix)
    forced = pd.DataFrame({"hub_degree": 10.0, "hub_threshold": 5.0,
                           "seasonal_return": 0.01}, index=index)
    start = strategy.warmup_bars(params)
    forced.iloc[start + 1:, forced.columns.get_loc("seasonal_return")] = -0.01
    signals = strategy.generate_signals(data, forced, None, params).data["signal"]
    assert signals.iloc[:start + 1].eq(0).all()
    assert signals.iloc[start + 1:start + 1 + params.hold_bars].eq(1).all()
    assert signals.iloc[start + 1 + params.hold_bars] == 0
    assert signals.iloc[start + 2 + params.hold_bars:start + 2 + 2 * params.hold_bars].eq(-1).all()
    assert set(signals.unique()).issubset({-1, 0, 1})


if __name__ == "__main__":
    _self_check()
