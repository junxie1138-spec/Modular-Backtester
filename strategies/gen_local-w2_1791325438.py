from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticMomentumParams:
    volatility_window: int = 40
    impulse_window: int = 12
    yield_band: float = 0.75
    dominance_threshold: float = 0.55
    min_upward_events: int = 3
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[PlasticMomentumParams]):
    strategy_id = "gen_local-w2_1791325438"

    @classmethod
    def params_type(cls) -> type[PlasticMomentumParams]:
        return PlasticMomentumParams

    @staticmethod
    def warmup_bars(params: PlasticMomentumParams) -> int:
        if params.volatility_window < 2 or params.impulse_window < 1:
            raise ValueError("Lookback windows must be positive; volatility_window must exceed one.")
        if params.yield_band <= 0 or not 0 < params.dominance_threshold < 1:
            raise ValueError("Invalid yield band or dominance threshold.")
        if not 1 <= params.min_upward_events <= params.impulse_window or params.hold_bars < 1:
            raise ValueError("Invalid event count or holding period.")
        return int(params.volatility_window + params.impulse_window + 1)

    def indicators(self, data: pd.DataFrame, params: PlasticMomentumParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["squared_return"] = out["return"].pow(2)
        out["prior_rms"] = out["squared_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).mean().pow(0.5).shift(1)
        out["standardized_return"] = (
            out["return"] / out["prior_rms"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["elastic_return"] = out["standardized_return"].clip(
            lower=-params.yield_band, upper=params.yield_band
        )
        out["plastic_return"] = out["standardized_return"] - out["elastic_return"]
        out["upward_impulse"] = out["plastic_return"].clip(lower=0.0)
        out["downward_impulse"] = -out["plastic_return"].clip(upper=0.0)
        out["upward_event"] = out["plastic_return"].gt(0.0).astype(float).where(
            out["plastic_return"].notna()
        )
        window = params.impulse_window
        out["positive_impulse"] = out["upward_impulse"].rolling(window, min_periods=window).sum()
        out["negative_impulse"] = out["downward_impulse"].rolling(window, min_periods=window).sum()
        out["upward_events"] = out["upward_event"].rolling(window, min_periods=window).sum()
        out["total_impulse"] = out["positive_impulse"] + out["negative_impulse"]
        out["plastic_dominance"] = (
            (out["positive_impulse"] - out["negative_impulse"])
            / out["total_impulse"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticMomentumParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            indicators["plastic_dominance"].ge(params.dominance_threshold)
            & indicators["upward_events"].ge(params.min_upward_events)
            & indicators["positive_impulse"].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        conviction = (
            (indicators["plastic_dominance"] - params.dominance_threshold)
            / (1.0 - params.dominance_threshold)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        expiry = -1
        entry_size = 0.25
        for i in range(len(data)):
            if expiry >= 0:
                if i == expiry:
                    expiry = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif eligible[i]:
                expiry = i + params.hold_bars
                entry_size = float(0.25 + 0.70 * conviction[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = PlasticMomentumParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = strategy.indicators(data, params)
    assert derived.index.equals(index)
    assert derived["plastic_dominance"].eq(0.0).all()
    artificial = pd.DataFrame(
        {"plastic_dominance": 0.8, "upward_events": 3.0, "positive_impulse": 1.0},
        index=index,
    )
    artificial.iloc[1:, artificial.columns.get_loc("plastic_dominance")] = 0.0
    result = strategy.generate_signals(data, artificial, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 11
    assert result["size"].gt(0.0).all()
    assert result["size"].iloc[1:9].nunique() == 1
    stronger = artificial.copy()
    stronger.iloc[0, stronger.columns.get_loc("plastic_dominance")] = 1.0
    stronger_result = strategy.generate_signals(data, stronger, None, params).data
    assert stronger_result["size"].iloc[1] > result["size"].iloc[1]


if __name__ == "__main__":
    _self_check()
