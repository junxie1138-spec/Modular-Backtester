from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalTrendParams:
    pattern_lag: int = 5
    distribution_window: int = 20
    smoothing_window: int = 5
    entry_strength: float = 0.06
    minimum_size: float = 0.20
    maximum_size: float = 0.95
    full_size_strength: float = 0.30


class GeneratedStrategy(BaseStrategy[OrdinalTrendParams]):
    strategy_id = "gen_local_1790986281"

    @classmethod
    def params_type(cls) -> type[OrdinalTrendParams]:
        return OrdinalTrendParams

    def warmup_bars(self, params: OrdinalTrendParams) -> int:
        lag = max(1, int(params.pattern_lag))
        window = max(2, int(params.distribution_window))
        smoothing = max(1, int(params.smoothing_window))
        return int(2 * lag + window + smoothing)

    def indicators(self, data: pd.DataFrame, params: OrdinalTrendParams) -> pd.DataFrame:
        lag = max(1, int(params.pattern_lag))
        window = max(2, int(params.distribution_window))
        smoothing = max(1, int(params.smoothing_window))
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        a = close.shift(2 * lag)
        b = close.shift(lag)
        c = close
        available = a.notna() & b.notna() & c.notna()

        # Labels give the chronological indices in ascending price order.
        patterns = {
            "012": (a < b) & (b < c),
            "021": (a < c) & (c < b),
            "102": (b < a) & (a < c),
            "120": (b < c) & (c < a),
            "201": (c < a) & (a < b),
            "210": (c < b) & (b < a),
        }
        valid = sum(mask.astype(float) for mask in patterns.values())
        coverage = valid.where(available).rolling(window, min_periods=window).mean()
        denominator = coverage.replace(0.0, np.nan)
        probabilities = pd.DataFrame(index=data.index)
        for label, mask in patterns.items():
            frequency = mask.astype(float).where(available).rolling(
                window, min_periods=window
            ).mean()
            probabilities["p_" + label] = frequency.div(denominator)

        entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
            axis=1, min_count=6
        ) / np.log(6.0)
        entropy = entropy.clip(0.0, 1.0)
        balance = probabilities["p_012"] - probabilities["p_210"]
        raw_strength = (balance * (1.0 - entropy)).where(coverage >= 0.80)
        strength = raw_strength.rolling(smoothing, min_periods=smoothing).mean()

        result = probabilities.fillna(0.0)
        result["pattern_coverage"] = coverage.fillna(0.0)
        result["ordinal_entropy"] = entropy.fillna(1.0)
        result["ordinal_balance"] = balance.fillna(0.0)
        result["raw_strength"] = raw_strength.fillna(0.0)
        result["strength"] = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalTrendParams,
    ) -> SignalFrame:
        strength = indicators["strength"].reindex(data.index).fillna(0.0)
        threshold = max(0.0, float(params.entry_strength))
        entry_condition = strength > threshold

        # Holding persists exactly while the entry condition remains true.
        # Its first false bar produces the sole exit condition.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)

        minimum = float(np.clip(params.minimum_size, 0.01, 1.0))
        maximum = float(np.clip(params.maximum_size, minimum, 1.0))
        scale = max(float(params.full_size_strength) - threshold, 1e-12)
        conviction = ((strength - threshold) / scale).clip(0.0, 1.0)
        decision_size = minimum + (maximum - minimum) * conviction
        df["size"] = decision_size.shift(1).fillna(minimum).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
