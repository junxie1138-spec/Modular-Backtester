from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalVolParams:
    complexity_window: int = 30
    baseline_lag: int = 5
    volatility_window: int = 20
    drift_window: int = 10
    holding_bars: int = 8
    entropy_drop: float = 0.025
    order_edge: float = 0.06
    target_volatility: float = 0.12
    min_size: float = 0.05
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[OrdinalVolParams]):
    strategy_id = "gen_local-w2_1790992133"

    @classmethod
    def params_type(cls) -> type[OrdinalVolParams]:
        return OrdinalVolParams

    @staticmethod
    def warmup_bars(params: OrdinalVolParams) -> int:
        return int(max(
            params.complexity_window + params.baseline_lag + 3,
            params.volatility_window + 1,
            params.drift_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: OrdinalVolParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["old_return"] = out["return"].shift(2)
        out["middle_return"] = out["return"].shift(1)
        a, b, c = out["old_return"], out["middle_return"], out["return"]
        # Stable time-order tie breaking gives exactly six possible permutations.
        valid = a.notna() & b.notna() & c.notna()
        out["pattern"] = pd.Series(np.select(
            [
                (a <= b) & (b <= c),
                (a <= c) & (c < b),
                (b < a) & (a <= c),
                (b <= c) & (c < a),
                (c < a) & (a <= b),
                (c < b) & (b < a),
            ],
            [0, 1, 2, 3, 4, 5],
            default=np.nan,
        ), index=data.index).where(valid)
        probability_names = []
        for k in range(6):
            name = f"probability_{k}"
            probability_names.append(name)
            out[name] = out["pattern"].eq(k).astype(float).where(valid).rolling(
                params.complexity_window, min_periods=params.complexity_window
            ).mean()
        probabilities = out[probability_names]
        out["entropy"] = -(
            probabilities * np.log(probabilities.clip(lower=1e-12))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        out["baseline_entropy"] = out["entropy"].shift(params.baseline_lag)
        out["entropy_contraction"] = out["baseline_entropy"] - out["entropy"]
        out["order_imbalance"] = out["probability_0"] - out["probability_5"]
        out["drift"] = out["return"].rolling(
            params.drift_window, min_periods=params.drift_window
        ).sum()
        out["absolute_motion"] = out["return"].abs().rolling(
            params.drift_window, min_periods=params.drift_window
        ).sum()
        out["efficiency"] = (
            out["drift"] / out["absolute_motion"].replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out["annual_volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["conviction"] = (
            (out["entropy_contraction"] / 0.15).clip(0.0, 1.0)
            * (out["order_imbalance"] / 0.30).clip(0.0, 1.0)
            * out["efficiency"]
        ).pow(1.0 / 3.0)
        out["target_size"] = (
            params.target_volatility
            / out["annual_volatility"].replace(0.0, np.nan)
            * (0.25 + 0.75 * out["conviction"])
        ).clip(lower=params.min_size, upper=params.max_size)
        out["entry_ready"] = (
            (out["entropy_contraction"] >= params.entropy_drop)
            & (out["order_imbalance"] >= params.order_edge)
            & (out["drift"] > 0.0)
            & (out["annual_volatility"] > 0.0)
            & out["target_size"].notna()
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalVolParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        targets = indicators["target_size"].to_numpy(dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        holding = max(1, int(params.holding_bars))
        for i in range(count):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                    sizes[i] = entry_size
                else:
                    # Reserve this decision bar for the mandatory timed exit.
                    exit_bar = -1
                continue
            if ready[i] and np.isfinite(targets[i]) and targets[i] > 0.0:
                entry_size = float(targets[i])
                exit_bar = i + holding
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
