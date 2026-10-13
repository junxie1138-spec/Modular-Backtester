from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietReversalParams:
    variance_ratio: float = 0.5
    drift_strength: float = 0.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[QuietReversalParams]):
    strategy_id = "gen_local_1790883914"

    @classmethod
    def params_type(cls):
        return QuietReversalParams

    def warmup_bars(self, params: QuietReversalParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: QuietReversalParams) -> pd.DataFrame:
        if not (0.0 < params.variance_ratio < 1.0):
            raise ValueError("variance_ratio must lie between zero and one")
        if not np.isfinite(params.drift_strength) or params.drift_strength <= 0.0:
            raise ValueError("drift_strength must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["recent_mean"] = out["returns"].rolling(4, min_periods=4).mean()
        out["prior_mean"] = out["returns"].shift(4).rolling(4, min_periods=4).mean()
        out["recent_variance"] = out["returns"].rolling(4, min_periods=4).var(ddof=1)
        out["prior_variance"] = out["returns"].shift(4).rolling(4, min_periods=4).var(ddof=1)
        out["variance_ratio"] = out["recent_variance"] / out["prior_variance"].clip(lower=1e-12)
        out["drift_strength"] = out["recent_mean"].abs() / np.sqrt(
            out["recent_variance"].clip(lower=1e-12)
        )
        out["qualified"] = (
            (out["prior_variance"] > 1e-12)
            & (out["variance_ratio"] < params.variance_ratio)
            & (out["drift_strength"] >= params.drift_strength)
            & (out["recent_mean"] * out["prior_mean"] < 0.0)
        ).fillna(False)
        out["candidate"] = np.where(
            out["qualified"], np.sign(out["recent_mean"].fillna(0.0)), 0
        ).astype(int)
        out["entry"] = out["candidate"].where(
            out["candidate"].ne(out["candidate"].shift(1).fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuietReversalParams,
    ) -> SignalFrame:
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = QuietReversalParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame({"entry": 0}, index=index)
    entries.iloc[[0, 3, 20], 0] = [1, -1, -1]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    result = strategy.generate_signals(data, entries, ctx, params).data
    expected = np.zeros(45, dtype=int)
    expected[1:19] = 1
    expected[21:39] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(data.index) and (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert not flat["entry"].any() and strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
