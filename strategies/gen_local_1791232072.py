from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    channel_bars: int = 40
    short_range_bars: int = 5
    baseline_bars: int = 20
    compression_ratio: float = 0.75
    maximum_position: float = 0.40
    efficiency_bars: int = 5
    minimum_efficiency: float = 0.30
    atr_bars: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791232072"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(params.channel_bars, params.short_range_bars,
                       params.baseline_bars, params.efficiency_bars + 1,
                       params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = data["high"] - data["low"]
        out["short_range"] = out["daily_range"].rolling(params.short_range_bars).mean()
        out["baseline_range"] = out["daily_range"].rolling(params.baseline_bars).mean()
        out["range_ratio"] = out["short_range"] / out["baseline_range"].replace(0.0, np.nan)
        out["channel_high"] = data["high"].rolling(params.channel_bars).max()
        out["channel_low"] = data["low"].rolling(params.channel_bars).min()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["channel_position"] = (data["close"] - out["channel_low"]) / out["channel_width"].replace(0.0, np.nan)
        out["close_change"] = data["close"].diff()
        out["path_length"] = out["close_change"].abs().rolling(params.efficiency_bars).sum()
        out["displacement"] = out["close_change"].rolling(params.efficiency_bars).sum()
        out["efficiency"] = out["displacement"] / out["path_length"].replace(0.0, np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["range_agrees"] = (
            out["range_ratio"].le(params.compression_ratio)
            & out["channel_position"].le(params.maximum_position)
            & out["channel_position"].ge(0.0)
        )
        out["efficiency_agrees"] = out["efficiency"].ge(params.minimum_efficiency)
        out["entry_ready"] = (
            out["range_agrees"] & out["efficiency_agrees"]
            & out["atr"].gt(0.0) & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        active = False
        fixed_stop = np.nan
        held_bars = 0
        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if not active and i > 0 and targets[i - 1] == 1:
                active = True
                fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                held_bars = 0
            if active:
                held_bars += 1
                if closes[i] < fixed_stop or held_bars >= params.holding_bars:
                    active = False
                    targets[i] = 0
                else:
                    targets[i] = 1
            elif ready[i] and (i == 0 or not ready[i - 1]):
                targets[i] = 1
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [101.0] * 8,
        "low": [97.0] * 8,
        "close": [100.0, 100.0, 98.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 8,
        "entry_ready": [True, False, False, True, False, False, False, False],
    }, index=index)
    params = CompressionParams(stop_atr=1.0, holding_bars=3)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    assert result.data["size"].gt(0.0).all()
    derived = GeneratedStrategy.indicators(data, CompressionParams())
    assert not derived["entry_ready"].any()
    assert GeneratedStrategy.warmup_bars(CompressionParams()) == 40


if __name__ == "__main__":
    _self_check()
