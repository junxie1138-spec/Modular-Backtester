from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAsymmetryParams:
    gap_window: int = 20
    reference_window: int = 252
    percentile: float = 0.85
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapAsymmetryParams]):
    strategy_id = "gen_local-w2_1791250450"

    @classmethod
    def params_type(cls) -> type[GapAsymmetryParams]:
        return GapAsymmetryParams

    def warmup_bars(self, params: GapAsymmetryParams) -> int:
        return int(params.gap_window + params.reference_window + 3)

    def indicators(self, data: pd.DataFrame, params: GapAsymmetryParams) -> pd.DataFrame:
        if params.gap_window < 2 or params.reference_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.percentile < 1.0 or params.holding_bars < 1:
            raise ValueError("Invalid percentile or holding period")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        valid = out["gap"].notna()
        out["positive_count"] = out["gap"].gt(0.0).astype(float).where(valid).rolling(params.gap_window).sum()
        out["negative_count"] = out["gap"].lt(0.0).astype(float).where(valid).rolling(params.gap_window).sum()
        out["positive_sum"] = out["gap"].clip(lower=0.0).rolling(params.gap_window).sum()
        out["negative_sum"] = (-out["gap"].clip(upper=0.0)).rolling(params.gap_window).sum()
        out["positive_mean"] = out["positive_sum"] / out["positive_count"].replace(0.0, np.nan)
        out["negative_mean"] = out["negative_sum"] / out["negative_count"].replace(0.0, np.nan)
        out["gap_rms"] = out["gap"].pow(2).rolling(params.gap_window).mean().pow(0.5)
        out["asymmetry"] = ((out["positive_mean"] - out["negative_mean"]) / out["gap_rms"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["absolute_asymmetry"] = out["asymmetry"].abs()
        out["threshold"] = out["absolute_asymmetry"].shift(1).rolling(params.reference_window).quantile(params.percentile)
        out["extreme"] = out["absolute_asymmetry"].gt(out["threshold"]) & out["threshold"].notna()
        out["entry_event"] = out["extreme"] & ~out["extreme"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapAsymmetryParams) -> SignalFrame:
        candidates = np.where(indicators["entry_event"], np.sign(indicators["asymmetry"].fillna(0.0)), 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(candidates):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                else:
                    signals[bar] = position
            elif candidate != 0:
                position = int(candidate)
                exit_bar = bar + params.holding_bars
                signals[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = GapAsymmetryParams(gap_window=2, reference_window=2, holding_bars=3)
    derived = pd.DataFrame({"entry_event": [True] * 10, "asymmetry": [1.0] * 4 + [-1.0] * 6}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, -1]
    assert (result["size"] > 0.0).all()
    flat = strategy.indicators(data, params)
    assert not flat["entry_event"].any()


if __name__ == "__main__":
    _self_check()
