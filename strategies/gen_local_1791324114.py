from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    lookback: int = 32
    entry_threshold: float = 0.40
    min_correlation: float = 0.20
    min_size: float = 0.20
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = "gen_local_1791324114"

    @classmethod
    def params_type(cls) -> type[RangeMigrationParams]:
        return RangeMigrationParams

    @staticmethod
    def warmup_bars(params: RangeMigrationParams) -> int:
        return int(params.lookback + 9)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        if params.lookback < 3:
            raise ValueError("lookback must be at least 3")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("entry_threshold must be finite and positive")
        if not 0 <= params.min_correlation <= 1:
            raise ValueError("min_correlation must be between zero and one")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("sizes must satisfy 0 < min_size <= max_size <= 1")

        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["width"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["range_budget"] = out["width"].rolling(4, min_periods=4).sum()
        out["migration"] = out["midpoint"].diff(4).div(out["range_budget"].replace(0.0, np.nan))
        out["lagged_migration"] = out["migration"].shift(4)
        n = params.lookback
        x = out["lagged_migration"].rolling(n, min_periods=n)
        y = out["migration"].rolling(n, min_periods=n)
        out["lag_mean"] = x.mean()
        out["migration_mean"] = y.mean()
        out["lag_variance"] = x.var(ddof=0)
        out["migration_std"] = y.std(ddof=0)
        out["covariance"] = y.cov(out["lagged_migration"], ddof=0)
        out["correlation"] = y.corr(out["lagged_migration"]).clip(-1.0, 1.0)
        out["slope"] = out["covariance"].div(out["lag_variance"].where(out["lag_variance"] > 1e-12))
        out["forecast"] = out["migration_mean"] + out["slope"] * (out["migration"] - out["lag_mean"])
        out["score"] = out["forecast"].div(out["migration_std"].where(out["migration_std"] > 1e-12))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeMigrationParams,
    ) -> SignalFrame:
        score = indicators["score"].replace([np.inf, -np.inf], np.nan)
        correlation = indicators["correlation"].replace([np.inf, -np.inf], np.nan)
        eligible = score.notna() & correlation.abs().ge(params.min_correlation)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & score.ge(params.entry_threshold)] = 1
        candidates.loc[eligible & score.le(-params.entry_threshold)] = -1
        strength = score.abs().fillna(0.0)
        entry_sizes = params.min_size + (params.max_size - params.min_size) * strength / (1.0 + strength)

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        held_size = params.min_size
        for i, candidate in enumerate(candidates.to_numpy()):
            if position == 0:
                if candidate != 0:
                    position = int(candidate)
                    held_size = float(entry_sizes.iloc[i])
            elif candidate == -position:
                # Exit flat on reversal; a later bar may establish the opposite position.
                position = 0
                held_size = params.min_size
            signals[i] = position
            sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeMigrationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [1.0, np.nan, -2.0, -2.0, 0.0, 1.0, 1.0], "correlation": 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert result["size"].iloc[4] > result["size"].iloc[1]
    assert result["size"].iloc[1] == result["size"].iloc[2]
    constant = pd.concat([data] * 10, ignore_index=True)
    constant.index = pd.date_range("2020-01-01", periods=len(constant))
    derived = GeneratedStrategy.indicators(constant, params)
    flat = GeneratedStrategy.generate_signals(constant, derived, None, params).data
    assert flat["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
