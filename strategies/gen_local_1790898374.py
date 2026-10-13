from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalFlowParams:
    recent_window: int = 18
    baseline_window: int = 54
    entropy_drop: float = 0.08
    minimum_drift: float = 0.0


class GeneratedStrategy(BaseStrategy[OrdinalFlowParams]):
    strategy_id = "gen_local_1790898374"

    @classmethod
    def params_type(cls) -> type[OrdinalFlowParams]:
        return OrdinalFlowParams

    @staticmethod
    def warmup_bars(params: OrdinalFlowParams) -> int:
        return int(max(3, params.recent_window) + max(3, params.baseline_window) + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalFlowParams) -> pd.DataFrame:
        recent = max(3, int(params.recent_window))
        baseline = max(3, int(params.baseline_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Equal closes use chronological precedence for deterministic ordering.
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c <= b),
            (b <= a) & (a <= c),
            (b <= c) & (c <= a),
            (c <= a) & (a <= b),
            (c <= b) & (b <= a),
        ]
        out["pattern"] = pd.Series(
            np.select(conditions, [0, 1, 2, 3, 4, 5], default=np.nan),
            index=data.index,
        ).where(valid)
        out["daily_return"] = close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        recent_entropy = pd.Series(0.0, index=data.index)
        baseline_entropy = pd.Series(0.0, index=data.index)
        for code in range(6):
            occurrence = out["pattern"].eq(code).astype(float).where(valid)
            short_probability = occurrence.rolling(recent, min_periods=recent).mean()
            old_probability = occurrence.shift(recent).rolling(
                baseline, min_periods=baseline
            ).mean()
            out[f"recent_probability_{code}"] = short_probability
            out[f"baseline_probability_{code}"] = old_probability
            recent_entropy -= short_probability * np.log(short_probability.clip(lower=1e-12))
            baseline_entropy -= old_probability * np.log(old_probability.clip(lower=1e-12))
        out["recent_entropy"] = recent_entropy / np.log(6.0)
        out["baseline_entropy"] = baseline_entropy / np.log(6.0)
        out["entropy_change"] = out["baseline_entropy"] - out["recent_entropy"]
        out["ordinal_balance"] = (
            out["recent_probability_0"] + out["recent_probability_1"]
            + out["recent_probability_2"] - out["recent_probability_3"]
            - out["recent_probability_4"] - out["recent_probability_5"]
        )
        out["drift"] = out["daily_return"].rolling(recent, min_periods=recent).mean()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalFlowParams,
    ) -> SignalFrame:
        required = ["entropy_change", "ordinal_balance", "drift"]
        ready = indicators[required].notna().all(axis=1)
        entry_condition = (
            ready
            & (indicators["entropy_change"] >= params.entropy_drop)
            & (indicators["ordinal_balance"] > 0.0)
            & (indicators["drift"] > params.minimum_drift)
        ).fillna(False)
        # Two consecutive true bars enter; the first false bar exits.
        confirmed = entry_condition & entry_condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
