from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    max_distance: float = 0.35
    min_edge: float = 0.0015
    profit_target: float = 0.02
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791270759"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["roc"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["scale"] = out["acceleration"].abs().rolling(2, min_periods=2).sum()
        scale = out["scale"].where(out["scale"] > 1e-12)
        out["shape_now"] = out["acceleration"] / scale
        out["shape_previous"] = out["acceleration"].shift(1) / scale
        for lag in (4, 5, 6):
            out[f"distance_{lag}"] = np.sqrt(
                (out["shape_now"] - out["shape_now"].shift(lag)) ** 2
                + (out["shape_previous"] - out["shape_previous"].shift(lag)) ** 2
            )
            out[f"outcome_{lag}"] = close.shift(lag - 2) / close.shift(lag) - 1.0
        distances = out[["distance_4", "distance_5", "distance_6"]].to_numpy()
        outcomes = out[["outcome_4", "outcome_5", "outcome_6"]].to_numpy()
        valid = np.isfinite(distances).all(axis=1) & np.isfinite(outcomes).all(axis=1)
        nearest = np.where(np.isfinite(distances), distances, np.inf).argmin(axis=1)
        rows = np.arange(len(out))
        out["match_distance"] = np.where(valid, distances[rows, nearest], np.nan)
        out["forecast"] = np.where(valid, outcomes[rows, nearest], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        if not (
            np.isfinite(params.max_distance) and params.max_distance > 0
            and np.isfinite(params.min_edge) and params.min_edge > 0
            and np.isfinite(params.profit_target) and params.profit_target > 0
            and 5 <= params.max_hold <= 10
            and int(params.max_hold) == params.max_hold
        ):
            raise ValueError("Invalid analog thresholds or holding period.")
        eligible = (
            indicators["match_distance"].le(params.max_distance)
            & indicators["forecast"].abs().ge(params.min_edge)
            & np.isfinite(indicators["forecast"])
        )
        setups = np.sign(indicators["forecast"].where(eligible, 0.0)).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # A decision made yesterday fills at today's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    age = 0
                age += 1
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                elif age >= params.max_hold or (
                    np.isfinite(closes[i])
                    and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                ):
                    # Close-based target; the flat decision fills next open.
                    position = 0
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = int(setups[i])
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams(max_hold=5)
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame({"match_distance": 0.0, "forecast": 0.0}, index=index)
    features.iloc[0, 1] = 0.01
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    signal = strategy.generate_signals(data, features, ctx, params).data["signal"]
    assert signal.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    for direction in (1, -1):
        features.iloc[0, 1] = direction * 0.01
        data.iloc[2, data.columns.get_loc("close")] = 100.0 + direction * 3.0
        signal = strategy.generate_signals(data, features, ctx, params).data["signal"]
        assert signal.tolist() == [0, direction, direction, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0 + np.array([0, 1, 0, 2, 1, 4, 2, 5, 4, 6, 3, 7])
    derived = strategy.indicators(data, params)
    assert derived["forecast"].iloc[:9].isna().all()
    pd.testing.assert_frame_equal(
        derived.iloc[:10], strategy.indicators(data.iloc[:10], params)
    )


if __name__ == "__main__":
    _self_check()
