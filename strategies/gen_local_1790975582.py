from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalElasticityParams:
    pattern_window: int = 24
    anchor_window: int = 15
    smoothing_window: int = 3
    complexity_floor: float = 0.72
    stretch_floor: float = 0.35


class GeneratedStrategy(BaseStrategy[OrdinalElasticityParams]):
    strategy_id = "gen_local_1790975582"

    @classmethod
    def params_type(cls) -> type[OrdinalElasticityParams]:
        return OrdinalElasticityParams

    @staticmethod
    def warmup_bars(params: OrdinalElasticityParams) -> int:
        return int(max(params.pattern_window + 2, params.anchor_window + params.smoothing_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalElasticityParams) -> pd.DataFrame:
        if params.pattern_window < 6 or params.anchor_window < 2 or params.smoothing_window < 1:
            raise ValueError("Invalid rolling window")
        if not 0.0 <= params.complexity_floor <= 1.0 or not np.isfinite(params.stretch_floor) or params.stretch_floor <= 0:
            raise ValueError("Invalid entry threshold")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["lag_one"] = close.shift(1)
        out["lag_two"] = close.shift(2)
        triples = np.column_stack([out["lag_two"], out["lag_one"], close])
        valid = np.isfinite(triples).all(axis=1)
        valid &= (triples[:, 0] != triples[:, 1]) & (triples[:, 0] != triples[:, 2]) & (triples[:, 1] != triples[:, 2])
        order = np.argsort(triples, axis=1, kind="stable")
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out["ordinal_code"] = pd.Series(codes, index=data.index).where(valid)
        probabilities = []
        for code in (5, 7, 11, 15, 19, 21):
            name = "probability_" + str(code)
            out[name] = out["ordinal_code"].eq(code).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            probabilities.append(name)
        p = out[probabilities].to_numpy()
        reversed_p = p[:, ::-1]
        midpoint = (p + reversed_p) / 2.0
        entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / np.log(6.0)
        forward_divergence = (p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12))).sum(axis=1)
        reverse_divergence = (reversed_p * np.log(np.maximum(reversed_p, 1e-12) / np.maximum(midpoint, 1e-12))).sum(axis=1)
        out["ordinal_entropy"] = entropy
        out["reversal_divergence"] = np.clip((forward_divergence + reverse_divergence) / (2.0 * np.log(2.0)), 0.0, 1.0)
        out["elastic_complexity"] = out["ordinal_entropy"] * (1.0 - np.sqrt(out["reversal_divergence"]))
        out["anchor"] = out["log_close"].rolling(params.anchor_window, min_periods=params.anchor_window).mean()
        out["return_volatility"] = out["log_return"].rolling(params.anchor_window, min_periods=params.anchor_window).std(ddof=0)
        denominator = (out["return_volatility"] * np.sqrt(params.anchor_window)).where(lambda x: x > 0)
        out["stretch"] = (out["log_close"] - out["anchor"]) / denominator
        out["smoothed_stretch"] = out["stretch"].rolling(params.smoothing_window, min_periods=params.smoothing_window).mean()
        out["eligible"] = (out["elastic_complexity"].ge(params.complexity_floor) & out["smoothed_stretch"].le(-params.stretch_floor)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalElasticityParams) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        # A false eligibility bar exits immediately and resets confirmation.
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame(index=index)
    features = pd.DataFrame({"eligible": [False, True, True, True, False, True, True]}, index=index)
    result = GeneratedStrategy.generate_signals(None, sample, features, None, OrdinalElasticityParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
