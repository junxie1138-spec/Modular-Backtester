from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalChangeParams:
    pattern_window: int = 4
    reference_window: int = 16
    percentile_window: int = 63
    change_percentile: float = 0.55


class GeneratedStrategy(BaseStrategy[OrdinalChangeParams]):
    strategy_id = "gen_local_1790966334"

    @classmethod
    def params_type(cls):
        return OrdinalChangeParams

    def warmup_bars(self, params: OrdinalChangeParams) -> int:
        return int(params.pattern_window + params.reference_window
                   + params.percentile_window + 4)

    def indicators(self, data: pd.DataFrame, params: OrdinalChangeParams) -> pd.DataFrame:
        if min(params.pattern_window, params.reference_window,
               params.percentile_window) < 3:
            raise ValueError("All windows must be at least three.")
        if not 0.0 < params.change_percentile < 1.0:
            raise ValueError("change_percentile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["lag1"] = out["return"].shift(1)
        out["lag2"] = out["return"].shift(2)
        valid = out[["return", "lag1", "lag2"]].notna().all(axis=1)
        # Stable chronological tie-breaking keeps flat returns well-defined.
        out["motif"] = (
            (out["lag2"] > out["lag1"]).astype(int)
            + 2 * (out["lag2"] > out["return"]).astype(int)
            + 4 * (out["lag1"] > out["return"]).astype(int)
        ).where(valid)
        out["complexity_change"] = 0.0
        for code in (0, 1, 3, 4, 6, 7):
            out[f"occurrence_{code}"] = (out["motif"] == code).astype(float).where(valid)
            out[f"recent_{code}"] = out[f"occurrence_{code}"].rolling(
                params.pattern_window, min_periods=params.pattern_window).mean()
            out[f"baseline_{code}"] = out[f"occurrence_{code}"].shift(
                params.pattern_window).rolling(
                    params.reference_window, min_periods=params.reference_window).mean()
            p = out[f"recent_{code}"]
            q = out[f"baseline_{code}"]
            out[f"mixture_{code}"] = (p + q) / 2.0
            m = out[f"mixture_{code}"]
            # Jensen-Shannon divergence measures ordinal distribution reorganization.
            out["complexity_change"] += (
                0.5 * p * np.log(p.where(p > 0, 1.0))
                + 0.5 * q * np.log(q.where(q > 0, 1.0))
                - m * np.log(m.where(m > 0, 1.0))
            )
        out["change_threshold"] = out["complexity_change"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.change_percentile)
        out["ordinal_tilt"] = (
            out["recent_0"] - out["recent_7"]
            - out["baseline_0"] + out["baseline_7"]
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: OrdinalChangeParams,
    ) -> SignalFrame:
        ready = indicators[[
            "complexity_change", "change_threshold", "ordinal_tilt"
        ]].notna().all(axis=1)
        changed = ready & (
            indicators["complexity_change"] > indicators["change_threshold"]
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[changed & (indicators["ordinal_tilt"] > 0)] = 1
        candidates.loc[changed & (indicators["ordinal_tilt"] < 0)] = -1
        # Opposite qualifying reorganization exits to flat; no timed exit.
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, candidate in enumerate(candidates.to_numpy()):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "complexity_change": np.ones(8),
        "change_threshold": np.full(8, 0.5),
        "ordinal_tilt": [0, 1, 1, 0, -1, -1, -1, 1],
    }, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalChangeParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
