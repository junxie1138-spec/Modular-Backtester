from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAbsorptionParams:
    ordinal_window: int = 12
    pressure_window: int = 8
    complexity_floor: float = 0.08
    contraction_share: float = 0.30
    pressure_floor: float = 0.10


class GeneratedStrategy(BaseStrategy[RangeAbsorptionParams]):
    strategy_id = "gen_local_1790997811"

    @classmethod
    def params_type(cls) -> type[RangeAbsorptionParams]:
        return RangeAbsorptionParams

    def warmup_bars(self, params: RangeAbsorptionParams) -> int:
        return int(max(params.ordinal_window + 2, params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeAbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        amplitude = (data["high"] - data["low"]).clip(lower=0.0)
        out["range"] = amplitude
        out["range_lag2"] = amplitude.shift(2)
        out["range_lag1"] = amplitude.shift(1)
        a = out["range_lag2"]
        b = out["range_lag1"]
        c = amplitude
        valid = a.notna() & b.notna() & c.notna()
        # Ties rank the earlier observation first, producing disjoint motifs.
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ]
        out["ordinal_pattern"] = pd.Series(
            np.select(conditions, [0, 1, 2, 3, 4, 5], default=np.nan),
            index=data.index,
            dtype=float,
        ).where(valid)
        probability_columns = []
        for motif in range(6):
            column = f"motif_probability_{motif}"
            occurrence = out["ordinal_pattern"].eq(motif).astype(float).where(valid)
            out[column] = occurrence.rolling(
                params.ordinal_window, min_periods=params.ordinal_window
            ).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        entropy_terms = probabilities * np.log(probabilities.clip(lower=1e-12))
        out["ordinal_entropy"] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["ordinal_disequilibrium"] = (
            probabilities.pow(2).sum(axis=1, min_count=6) - 1.0 / 6.0
        ) / (5.0 / 6.0)
        out["ordinal_complexity"] = (
            out["ordinal_entropy"] * out["ordinal_disequilibrium"]
        ).clip(lower=0.0, upper=1.0)
        out["contraction_share"] = out["motif_probability_5"]
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / amplitude.replace(0.0, np.nan)
        ).clip(lower=-1.0, upper=1.0)
        out.loc[amplitude.eq(0.0), "close_location"] = 0.0
        out["usable_volume"] = data["volume"].clip(lower=0.0)
        out["signed_closing_volume"] = out["close_location"] * out["usable_volume"]
        out["pressure_numerator"] = out["signed_closing_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure_denominator"] = out["usable_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["closing_pressure"] = (
            out["pressure_numerator"]
            / out["pressure_denominator"].replace(0.0, np.nan)
        )
        out["window_return"] = (
            data["close"]
            / data["close"].shift(params.pressure_window).replace(0.0, np.nan)
            - 1.0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeAbsorptionParams,
    ) -> SignalFrame:
        required = [
            "ordinal_complexity", "contraction_share", "closing_pressure", "window_return"
        ]
        ready = indicators[required].notna().all(axis=1)
        organized_contraction = (
            indicators["ordinal_complexity"].ge(params.complexity_floor)
            & indicators["contraction_share"].ge(params.contraction_share)
        )
        absorption_divergence = (
            indicators["closing_pressure"].gt(params.pressure_floor)
            & indicators["window_return"].lt(0.0)
        )
        entry_condition = ready & organized_contraction & absorption_divergence
        # Target remains long only while both primitives agree. A false
        # entry predicate is the sole exit; there is no timed or price exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
