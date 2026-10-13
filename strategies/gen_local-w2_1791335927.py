from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BottleneckParams:
    flow_bars: int = 5
    rank_bars: int = 120
    volume_percentile: float = 0.80
    progress_percentile: float = 0.25
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[BottleneckParams]):
    strategy_id = "gen_local-w2_1791335927"

    @classmethod
    def params_type(cls) -> type[BottleneckParams]:
        return BottleneckParams

    @staticmethod
    def warmup_bars(params: BottleneckParams) -> int:
        if params.flow_bars < 1 or params.rank_bars < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; rank_bars must exceed one.")
        if not 0.0 < params.volume_percentile <= 1.0:
            raise ValueError("volume_percentile must be in (0, 1].")
        if not 0.0 < params.progress_percentile <= 1.0:
            raise ValueError("progress_percentile must be in (0, 1].")
        return int(max(200, params.flow_bars + params.rank_bars))

    @staticmethod
    def indicators(data: pd.DataFrame, params: BottleneckParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["daily_range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["range_budget"] = out["daily_range"].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out["displacement"] = data["close"].diff(params.flow_bars)
        out["progress"] = out["displacement"].abs().div(out["range_budget"].where(out["range_budget"] > 0))
        out["volume_mean"] = data["volume"].where(data["volume"] >= 0).rolling(params.flow_bars, min_periods=params.flow_bars).mean()
        out["volume_rank"] = out["volume_mean"].rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        out["progress_rank"] = out["progress"].rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BottleneckParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        valid = indicators[["ma200", "displacement", "volume_rank", "progress_rank"]].notna().all(axis=1)
        bottleneck = valid & (indicators["volume_rank"] >= params.volume_percentile) & (indicators["progress_rank"] <= params.progress_percentile)
        long_entry = bottleneck & (data["close"] > indicators["ma200"]) & (indicators["displacement"] < 0)
        short_entry = bottleneck & (data["close"] < indicators["ma200"]) & (indicators["displacement"] > 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position and i == exit_bar:
                position = 0
            elif not position and candidate:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = BottleneckParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    close = np.array([101, 99, 99, 99, 99, 101, 101, 101, 101, 99, 99, 99], dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"ma200": 100.0, "displacement": np.where(close > 100, -1.0, 1.0), "volume_rank": 0.95, "progress_rank": 0.10}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1]
    assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) == 200
    indicators.loc[:, "volume_rank"] = np.nan
    assert GeneratedStrategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
