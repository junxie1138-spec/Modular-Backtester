from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 21
    entropy_gap: float = 0.05
    seasonal_return_floor: float = 0.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791210635"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # Covers the twelve-calendar-month lag plus ordinal-window history.
        return 400 + int(params.complexity_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.complexity_window < 3:
            raise ValueError("complexity_window must be at least 3")
        if not np.isfinite(params.entropy_gap) or params.entropy_gap < 0:
            raise ValueError("entropy_gap must be finite and nonnegative")
        if not np.isfinite(params.seasonal_return_floor):
            raise ValueError("seasonal_return_floor must be finite")

        out = pd.DataFrame(index=data.index)
        if data.empty:
            for name in ("ordinal_code", "ordinal_entropy", "incoming_entropy", "historical_entropy", "historical_return"):
                out[name] = pd.Series(index=data.index, dtype=float)
            return out

        close = data["close"].astype(float)
        triples = np.column_stack((close.shift(2), close.shift(1), close))
        valid = np.isfinite(triples).all(axis=1) & (triples > 0).all(axis=1)
        # Stable sorting resolves equal closes by their chronological order.
        order = np.argsort(triples, axis=1, kind="stable")
        codes = pd.Series(9 * order[:, 0] + 3 * order[:, 1] + order[:, 2], index=data.index, dtype=float).where(valid)
        out["ordinal_code"] = codes

        probabilities = pd.DataFrame(index=data.index)
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = codes.eq(code).astype(float).where(valid)
            probabilities[code] = occurrence.rolling(params.complexity_window, min_periods=params.complexity_window).mean()
        entropy_terms = probabilities * np.log(probabilities.clip(lower=1e-12))
        out["ordinal_entropy"] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)

        months = data.index.to_period("M")
        monthly = pd.DataFrame({
            "first_open": data["open"].groupby(months).first(),
            "last_close": close.groupby(months).last(),
            "entropy": out["ordinal_entropy"].groupby(months).last(),
        })
        monthly = monthly.reindex(pd.period_range(months.min(), months.max(), freq="M"))
        monthly["return"] = monthly["last_close"] / monthly["first_open"].where(monthly["first_open"] > 0) - 1.0
        # The dataset's first month may be incomplete.
        monthly.loc[monthly.index[0], ["return", "entropy"]] = np.nan

        # Every mapped value was already known before the current month began.
        out["incoming_entropy"] = monthly["entropy"].shift(1).reindex(months).to_numpy()
        out["historical_entropy"] = monthly["entropy"].shift(12).reindex(months).to_numpy()
        out["historical_return"] = monthly["return"].shift(12).reindex(months).to_numpy()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        needed = indicators[["incoming_entropy", "historical_entropy", "historical_return"]]
        eligible = (
            needed.notna().all(axis=1)
            & (indicators["historical_return"] > params.seasonal_return_floor)
            & (indicators["incoming_entropy"] + params.entropy_gap < indicators["historical_entropy"])
        ).fillna(False)
        # Two consecutive eligible closes enter; any eligibility reversal exits.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    dates = pd.bdate_range("2018-01-01", periods=900)
    t = np.arange(len(dates), dtype=float)
    close = 100.0 * np.exp(0.0003 * t + 0.025 * np.sin(t / 13.0))
    data = pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=dates)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:777], params)
    pd.testing.assert_frame_equal(full.iloc[:777], prefix)
    fixture = pd.DataFrame({"incoming_entropy": 0.4, "historical_entropy": 0.8, "historical_return": [-0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1]}, index=dates[:7])
    result = GeneratedStrategy.generate_signals(data.iloc[:7], fixture, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 1]
    assert (result["size"] > 0).all()
