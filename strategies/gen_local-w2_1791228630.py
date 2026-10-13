from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    tension_window: int = 4
    coupling_window: int = 12
    rank_window: int = 63
    entry_percentile: float = 0.8
    regime_percentile: float = 0.7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791228630"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.rank_window + max(params.tension_window, params.coupling_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.tension_window, params.coupling_window, params.rank_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0.5 < params.entry_percentile < 1 and 0.5 < params.regime_percentile < 1):
            raise ValueError("Percentiles must lie strictly between 0.5 and 1.")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out["tension"] = out["gap"].rolling(params.tension_window).sum()
        out["tension_rank"] = out["tension"].rolling(params.rank_window).rank(pct=True)
        out["coupling"] = out["gap"].rolling(params.coupling_window).corr(out["intraday"]).replace([np.inf, -np.inf], np.nan)
        out["coupling_rank"] = out["coupling"].rolling(params.rank_window).rank(pct=True)
        out["regime"] = pd.Series(
            np.select(
                [out["coupling_rank"] >= params.regime_percentile,
                 out["coupling_rank"] <= 1.0 - params.regime_percentile],
                [1.0, -1.0], default=np.nan),
            index=data.index,
        ).ffill().fillna(0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        rank = indicators["tension_rank"]
        stretch = pd.Series(
            np.select(
                [rank >= params.entry_percentile, rank <= 1.0 - params.entry_percentile],
                [1, -1], default=0),
            index=data.index,
        )
        valid = indicators[["tension_rank", "coupling_rank"]].notna().all(axis=1)
        entry = (stretch * indicators["regime"]).where(valid, 0).fillna(0).astype(int)
        position = 0
        positions = np.zeros(len(data), dtype=int)
        for i, desired in enumerate(entry.to_numpy()):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"tension_rank": [0.9, 0.5, 0.1, 0.1, 0.5, 0.9, np.nan],
         "coupling_rank": 0.9, "regime": 1.0}, index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=180)
    t = np.arange(180, dtype=float)
    closing = 100.0 * np.exp(0.002 * t + 0.025 * np.sin(t / 3.0))
    opening = closing * np.exp(0.007 * np.sin(t / 2.0))
    data = pd.DataFrame(
        {"open": opening, "high": np.maximum(opening, closing) * 1.01,
         "low": np.minimum(opening, closing) * 0.99, "close": closing, "volume": 1000.0},
        index=index,
    )
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:120], strategy.indicators(data.iloc[:120], params))
    signals = strategy.generate_signals(data, full, ctx, params).data
    pd.testing.assert_frame_equal(
        signals.iloc[:120],
        strategy.generate_signals(data.iloc[:120], full.iloc[:120], ctx, params).data,
    )
    assert signals["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()


if __name__ == "__main__":
    _self_check()
