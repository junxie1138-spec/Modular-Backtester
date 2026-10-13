from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    history_bars: int = 252
    rank_bars: int = 126
    tail_percentile: float = 0.8
    hold_bars: int = 8
    min_samples: int = 10


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791284263"

    @classmethod
    def params_type(cls):
        return SeasonalParams

    @staticmethod
    def warmup_bars(params: SeasonalParams) -> int:
        if min(params.history_bars, params.rank_bars, params.hold_bars, params.min_samples) < 1:
            raise ValueError("Lookbacks, holding period and sample count must be positive.")
        if not 0.5 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1.")
        return int(params.history_bars + params.rank_bars + params.hold_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        days = data.index.day
        edge = (days <= 7) | (days >= 24)
        out["calendar_phase"] = data.index.dayofweek + 5 * edge.astype(int)
        # A decision at t enters at open[t+1], exits at open[t+hold+1].
        # Only completed open-to-open outcomes are used to learn seasonality.
        out["matured_phase"] = out["calendar_phase"].shift(params.hold_bars + 1)
        opening = data["open"].where(data["open"] > 0)
        out["matured_return"] = (opening / opening.shift(params.hold_bars) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        out["forecast"] = np.nan
        for phase in range(10):
            observed = out["matured_phase"].eq(phase) & out["matured_return"].notna()
            count = observed.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            total = out["matured_return"].where(observed, 0.0).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum()
            mean = total / count.where(count >= params.min_samples)
            out[f"sample_count_{phase}"] = count
            out[f"return_sum_{phase}"] = total
            out[f"mean_return_{phase}"] = mean
            current = out["calendar_phase"].eq(phase)
            out.loc[current, "forecast"] = mean.loc[current]
        out["percentile"] = out["forecast"].rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).rank(pct=True)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalParams,
    ) -> SignalFrame:
        warmup = GeneratedStrategy.warmup_bars(params)
        long_entry = (
            indicators["percentile"].ge(params.tail_percentile)
            & indicators["forecast"].gt(0.0)
        )
        short_entry = (
            indicators["percentile"].le(1.0 - params.tail_percentile)
            & indicators["forecast"].lt(0.0)
        )
        candidates = np.select(
            [long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0
        )
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(warmup, len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signal[i] = position
                continue
            if candidates[i]:
                position = int(candidates[i])
                exit_bar = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SeasonalParams()
    start = GeneratedStrategy.warmup_bars(params)
    index = pd.bdate_range("2020-01-01", periods=start + 30)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    for direction in (1, -1):
        ind = pd.DataFrame(
            {"forecast": float(direction), "percentile": 0.99 if direction == 1 else 0.01},
            index=index,
        )
        # Opposite evidence during the hold must neither reverse nor close it.
        ind.iloc[start + 1:start + params.hold_bars] = [
            -float(direction), 0.01 if direction == 1 else 0.99
        ]
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result["signal"].iloc[:start + 1].eq(0).all()
        assert result["signal"].iloc[start + 1:start + params.hold_bars + 1].eq(direction).all()
        assert result["signal"].iloc[start + params.hold_bars + 1] == 0
        assert result["size"].gt(0).all()
    actual = GeneratedStrategy.indicators(data, params)
    assert actual["forecast"].dropna().eq(0.0).all()
    assert GeneratedStrategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
