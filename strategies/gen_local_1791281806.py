from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    compression_bars: int = 10
    rank_window: int = 126
    compression_percentile: float = 0.20
    crowding_percentile: float = 0.80
    pressure_threshold: float = 0.15
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791281806"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        if params.compression_bars < 2 or params.rank_window < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and hold_bars must be positive.")
        if not (0 < params.compression_percentile < 1 and 0 < params.crowding_percentile < 1):
            raise ValueError("Percentile thresholds must lie strictly between zero and one.")
        if not 0 <= params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must lie in [0, 1).")
        return max(200, params.compression_bars + params.rank_window - 1)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        n = params.compression_bars
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["daily_range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["range_fraction"] = out["daily_range"] / close
        out["envelope_high"] = data["high"].rolling(n, min_periods=n).max()
        out["envelope_low"] = data["low"].rolling(n, min_periods=n).min()
        out["envelope_fraction"] = (out["envelope_high"] - out["envelope_low"]) / close
        out["compression_rank"] = out["envelope_fraction"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["volume_density"] = volume / out["range_fraction"]
        out["mean_density"] = out["volume_density"].rolling(n, min_periods=n).mean()
        out["crowding_rank"] = out["mean_density"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["close_location"] = (
            (2 * close - data["high"] - data["low"]) / out["daily_range"]
        ).clip(-1, 1)
        out["signed_volume"] = volume * out["close_location"]
        out["volume_sum"] = volume.rolling(n, min_periods=n).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(n, min_periods=n).sum()
        out["pressure"] = out["signed_volume_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CompressionParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        required = ["compression_rank", "crowding_rank", "pressure", "ma200"]
        valid = indicators[required].notna().all(axis=1) & data["close"].notna()
        setup = (
            valid
            & (indicators["compression_rank"] <= params.compression_percentile)
            & (indicators["crowding_rank"] >= params.crowding_percentile)
        )
        long_entry = setup & (data["close"] > indicators["ma200"]) & (
            indicators["pressure"] > params.pressure_threshold
        )
        short_entry = setup & (data["close"] < indicators["ma200"]) & (
            indicators["pressure"] < -params.pressure_threshold
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                signal[i] = position
            elif entries[i]:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = CompressionParams(hold_bars=4)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"compression_rank": 0.1, "crowding_rank": 0.9, "pressure": -0.5, "ma200": 110.0},
        index=index,
    )
    indicators.iloc[:2, indicators.columns.get_loc("pressure")] = 0.5
    indicators.iloc[:2, indicators.columns.get_loc("ma200")] = 90.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    indicators.loc[:, "compression_rank"] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(CompressionParams(compression_bars=80, rank_window=160)) == 239


if __name__ == "__main__":
    _self_check()
