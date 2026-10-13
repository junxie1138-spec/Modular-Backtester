from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecoveryParams:
    drawdown_window: int = 20
    complexity_window: int = 4
    min_drawdown: float = 0.01


class GeneratedStrategy(BaseStrategy[OrdinalRecoveryParams]):
    strategy_id = "gen_local_1791239248"

    @classmethod
    def params_type(cls) -> type[OrdinalRecoveryParams]:
        return OrdinalRecoveryParams

    def warmup_bars(self, params: OrdinalRecoveryParams) -> int:
        return max(params.drawdown_window, 2 * params.complexity_window + 3, 4)

    def indicators(self, data: pd.DataFrame, params: OrdinalRecoveryParams) -> pd.DataFrame:
        if params.drawdown_window < 4 or params.complexity_window < 3:
            raise ValueError("Lookback windows must be at least four and three bars respectively")
        if not np.isfinite(params.min_drawdown) or not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("min_drawdown must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        a, b, c = close.shift(2), close.shift(1), close
        out["ordinal_pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan,
        )
        out["previous_pattern"] = out["ordinal_pattern"].shift(1)
        out["transition"] = 6 * out["previous_pattern"] + out["ordinal_pattern"]
        valid = out["transition"].notna()
        window = params.complexity_window
        transition_counts = pd.DataFrame({
            code: out["transition"].eq(code).astype(float).where(valid).rolling(window).sum()
            for code in range(36)
        }, index=data.index)
        source_counts = pd.DataFrame({
            code: out["previous_pattern"].eq(code).astype(float).where(valid).rolling(window).sum()
            for code in range(6)
        }, index=data.index)
        ready = valid.astype(int).rolling(window).sum().eq(window)
        out["distinct_transitions"] = transition_counts.gt(0).sum(axis=1).where(ready)
        out["distinct_sources"] = source_counts.gt(0).sum(axis=1).where(ready)
        out["branch_complexity"] = out["distinct_transitions"] / out["distinct_sources"].replace(0, np.nan)
        out["prior_complexity"] = out["branch_complexity"].shift(window)
        out["complexity_expansion"] = out["branch_complexity"] - out["prior_complexity"]
        out["rolling_peak"] = close.rolling(params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["rolling_peak"]
        out["recovery"] = close / close.shift(3) - 1.0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalRecoveryParams) -> SignalFrame:
        required = indicators[["drawdown", "recovery", "complexity_expansion"]]
        eligible = (np.isfinite(required).all(axis=1)
                    & indicators["drawdown"].ge(params.min_drawdown)
                    & indicators["recovery"].gt(0)
                    & indicators["complexity_expansion"].gt(0))
        df = pd.DataFrame(index=data.index)
        # The same predicate governs both sides: enter when true, exit when false.
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = OrdinalRecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"close": [100.0, 99.0, 100.0, 99.0]}, index=index)
    derived = pd.DataFrame({"drawdown": [np.nan, 0.02, 0.02, 0.02],
                            "recovery": [np.nan, 0.01, 0.01, -0.01],
                            "complexity_expansion": [np.nan, 0.5, -0.5, 0.5]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0]
    assert result["size"].gt(0).all()
    values = np.random.default_rng(7).normal(0, 0.01, 100)
    close = pd.Series(100 * np.exp(np.cumsum(values)), index=pd.date_range("2020-01-01", periods=100))
    sample = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 1000.0})
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:70], params)
    pd.testing.assert_frame_equal(full.iloc[:70], prefix)
    assert full["prior_complexity"].first_valid_index() == sample.index[2 * params.complexity_window + 2]
    assert strategy.warmup_bars(params) >= 2 * params.complexity_window + 3


if __name__ == "__main__":
    _self_check()
