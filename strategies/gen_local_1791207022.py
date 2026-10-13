from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PocketParams:
    ma_window: int = 24
    fast_range: int = 3
    slow_range: int = 20
    compression: float = 0.72
    entry_z: float = 1.15
    min_overlap: float = 0.60
    hold_bars: int = 4
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[PocketParams]):
    strategy_id = "gen_local_1791207022"

    @classmethod
    def params_type(cls):
        return PocketParams

    def warmup_bars(self, params: PocketParams) -> int:
        if (
            params.ma_window < 2
            or params.fast_range < 1
            or params.slow_range < params.fast_range
            or params.hold_bars < 1
            or not np.isfinite([params.compression, params.entry_z, params.min_overlap, params.size]).all()
            or not 0 < params.compression < 1
            or params.entry_z <= 0
            or not 0 <= params.min_overlap <= 1
            or params.size <= 0
        ):
            raise ValueError("Invalid pocket strategy parameters")
        return int(max(params.ma_window, params.slow_range + 1, 2) + 1)

    def indicators(self, data: pd.DataFrame, params: PocketParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].where(out["std"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["relative_range"] = out["daily_range"] / close.where(close > 0)
        out["fast_range"] = out["relative_range"].rolling(params.fast_range).mean()
        out["baseline_range"] = out["relative_range"].rolling(params.slow_range).mean().shift(1)
        out["compression_ratio"] = out["fast_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        out["overlap_width"] = (
            data["high"].rolling(2).min() - data["low"].rolling(2).max()
        ).clip(lower=0)
        out["smaller_range"] = out["daily_range"].rolling(2).min()
        out["overlap_ratio"] = out["overlap_width"] / out["smaller_range"].where(out["smaller_range"] > 0)
        eligible = (
            out["compression_ratio"].le(params.compression)
            & out["overlap_ratio"].ge(params.min_overlap)
            & out["z"].notna()
        )
        out["setup"] = np.select(
            [eligible & out["z"].le(-params.entry_z),
             eligible & out["z"].ge(params.entry_z)],
            [1, -1], default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PocketParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        setup = indicators["setup"].fillna(0).astype(int)
        confirmed = setup.where(setup.eq(setup.shift(1)), 0).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(confirmed):
            if position:
                if i == exit_bar:
                    position = 0
                raw[i] = position
                continue
            if entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = PocketParams()
    index = pd.date_range("2020-01-01", periods=15)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"setup": [0, 1, 1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
