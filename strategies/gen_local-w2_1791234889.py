from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeFoldParams:
    window: int = 18
    contraction_window: int = 3
    entry_threshold: float = 0.8
    max_contraction: float = 0.9
    min_overlap: float = 0.55
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeFoldParams]):
    strategy_id = "gen_local-w2_1791234889"

    @classmethod
    def params_type(cls):
        return RangeFoldParams

    def warmup_bars(self, params: RangeFoldParams) -> int:
        return int(max(params.window, params.contraction_window))

    def indicators(self, data: pd.DataFrame, params: RangeFoldParams) -> pd.DataFrame:
        if not (params.window >= 2 and 1 <= params.contraction_window <= params.window):
            raise ValueError("Require window >= 2 and 1 <= contraction_window <= window")
        if not (params.entry_threshold > 0 and params.max_contraction > 0
                and 0 <= params.min_overlap <= 1
                and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid thresholds or position sizes")
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["midpoint"] = (data["high"] + data["low"]) / 2
        out["mean_range"] = out["daily_range"].rolling(params.window).mean()
        out["range_sum"] = out["daily_range"].rolling(params.window).sum()
        out["envelope"] = (data["high"].rolling(params.window).max()
                           - data["low"].rolling(params.window).min())
        out["anchor"] = out["midpoint"].rolling(params.window).mean()
        out["short_range"] = out["daily_range"].rolling(params.contraction_window).mean()
        out["overlap"] = (1 - out["envelope"] / out["range_sum"].replace(0, np.nan)).clip(0, 1)
        out["contraction"] = out["short_range"] / out["mean_range"].replace(0, np.nan)
        out["stretch"] = (data["close"] - out["anchor"]) / out["mean_range"].replace(0, np.nan)
        out["score"] = -out["stretch"] * out["overlap"] / out["contraction"].clip(lower=0.5)
        out["ready"] = (out["score"].notna() & np.isfinite(out["score"])
                        & data[["high", "low", "close"]].notna().all(axis=1))
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeFoldParams,
    ) -> SignalFrame:
        eligible = (indicators["ready"].astype(bool)
                    & (indicators["overlap"] >= params.min_overlap)
                    & (indicators["contraction"] <= params.max_contraction))
        score = indicators["score"].replace([np.inf, -np.inf], np.nan).fillna(0)
        entries = np.where(eligible & (score >= params.entry_threshold), 1,
                           np.where(eligible & (score <= -params.entry_threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        held_size = params.min_size
        for i, entry in enumerate(entries):
            if position == 0 and entry != 0:
                position = int(entry)
                strength = abs(float(score.iloc[i])) / params.entry_threshold
                held_size = float(np.clip(
                    params.min_size + (params.max_size - params.min_size) * (strength - 1),
                    params.min_size, params.max_size,
                ))
            elif position != 0 and entry == -position:
                # Only an opposite qualifying entry closes the position.
                position = 0
            signals[i] = position
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeFoldParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=18)
    indicators = pd.DataFrame(
        {"ready": True, "overlap": 0.8, "contraction": 0.7,
         "score": [0.8, 0.0, -1.6, -1.6, 0.0, 1.6, 1.6, 0.0]},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert np.isclose(result["size"].iloc[1], params.min_size)
    assert np.isclose(result["size"].iloc[4], params.max_size)
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert np.isfinite(flat.to_numpy(dtype=float)).all()
    assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
