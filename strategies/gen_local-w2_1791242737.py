from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    entropy_window: int = 12
    entropy_ceiling: float = 0.90
    min_entropy_drop: float = 0.01


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791242737"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(params.entropy_window) + 4

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if params.entropy_window < 3 or not 0 < params.entropy_ceiling <= 1 or not 0 <= params.min_entropy_drop <= 1:
            raise ValueError("Invalid ordinal entropy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["oldest_return"] = out["return"].shift(2)
        a, b, c = out["oldest_return"], out["previous_return"], out["return"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns are ranked chronologically, earlier observations first.
        out["pattern"] = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"probability_{code}"
            out[name] = out["pattern"].eq(code).astype(float).where(valid).rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        terms = probabilities * np.log(probabilities.where(probabilities > 0))
        out["entropy"] = (-terms.fillna(0).sum(axis=1) / np.log(6.0)).where(probabilities.notna().all(axis=1))
        out["entropy_drop"] = -out["entropy"].diff()
        out["confidence"] = ((params.entropy_ceiling - out["entropy"]) / params.entropy_ceiling + out["probability_0"]).clip(0, 1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        entry = (
            indicators["entropy"].lt(params.entropy_ceiling)
            & indicators["entropy_drop"].ge(params.min_entropy_drop)
            & indicators["return"].gt(indicators["previous_return"])
            & indicators["previous_return"].gt(0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The same eligibility predicate maintains the position and exits on reversal.
        df["signal"] = entry.astype(int)
        df["size"] = (0.20 + 0.75 * indicators["confidence"].fillna(0).clip(0, 1)).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), OrdinalParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "entropy": [np.nan, 0.8, 0.7, 0.8, 0.8],
        "entropy_drop": [np.nan, 0.02, 0.1, -0.1, 0.0],
        "return": [np.nan, 0.02, 0.03, 0.04, 0.05],
        "previous_return": [np.nan, 0.01, 0.02, 0.03, 0.04],
        "confidence": [np.nan, 0.2, 0.6, 0.1, 0.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert np.allclose(result["size"], [0.2, 0.2, 0.35, 0.65, 0.275])
    flat = pd.concat([data] * 8, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat))
    computed = strategy.indicators(flat, params)
    assert computed["entropy_drop"].first_valid_index() == flat.index[strategy.warmup_bars(params) - 1]
    assert strategy.generate_signals(flat, computed, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
