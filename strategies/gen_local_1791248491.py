from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnvelopeParams:
    envelope_bars: int = 5
    rank_bars: int = 63
    entry_percentile: float = 0.80
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[EnvelopeParams]):
    strategy_id = "gen_local_1791248491"

    @classmethod
    def params_type(cls) -> type[EnvelopeParams]:
        return EnvelopeParams

    def warmup_bars(self, params: EnvelopeParams) -> int:
        return int(max(200, params.atr_bars + 1,
                       2 * params.envelope_bars + params.rank_bars))

    def indicators(self, data: pd.DataFrame, params: EnvelopeParams) -> pd.DataFrame:
        if (min(params.envelope_bars, params.rank_bars, params.atr_bars,
                params.max_hold_bars) < 1 or params.trail_atr <= 0
                or not 0.5 < params.entry_percentile < 1):
            raise ValueError("Invalid envelope strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ma200"] = close.rolling(200).mean()
        out["floor"] = close.rolling(params.envelope_bars).min()
        out["ceiling"] = close.rolling(params.envelope_bars).max()
        denominator = out["atr"].where(out["atr"] > 0)
        out["floor_strength"] = out["floor"].diff(params.envelope_bars) / denominator
        out["ceiling_strength"] = -out["ceiling"].diff(params.envelope_bars) / denominator
        out["floor_rank"] = out["floor_strength"].rolling(params.rank_bars).rank(pct=True)
        out["ceiling_rank"] = out["ceiling_strength"].rolling(params.rank_bars).rank(pct=True)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: EnvelopeParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        ready = np.isfinite(indicators[[
            "atr", "ma200", "floor_strength", "ceiling_strength",
            "floor_rank", "ceiling_rank"
        ]].to_numpy(dtype=float)).all(axis=1)
        long_entry = (
            (indicators["floor_rank"] >= params.entry_percentile)
            & (indicators["floor_rank"].shift(1).fillna(0) < params.entry_percentile)
            & (indicators["floor_strength"] > 0)
        ).to_numpy()
        short_entry = (
            (indicators["ceiling_rank"] >= params.entry_percentile)
            & (indicators["ceiling_rank"].shift(1).fillna(0) < params.entry_percentile)
            & (indicators["ceiling_strength"] > 0)
        ).to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        watermark, stop = np.nan, np.nan
        for i in range(len(data)):
            if not ready[i] or atr[i] <= 0 or not np.isfinite(close[i]):
                position, age = 0, 0
                continue
            if position:
                # This bar is held: yesterday's decision filled at today's open.
                if age == 0:
                    if i == 0 or not np.isfinite(opening[i]) or not np.isfinite(atr[i - 1]):
                        position = 0
                        continue
                    watermark = close[i]
                    stop = opening[i] - position * params.trail_atr * atr[i - 1]
                age += 1
                if position == 1:
                    watermark = max(watermark, close[i])
                    stop = max(stop, watermark - params.trail_atr * atr[i])
                    exit_now = close[i] <= stop or close[i] <= ma[i]
                else:
                    watermark = min(watermark, close[i])
                    stop = min(stop, watermark + params.trail_atr * atr[i])
                    exit_now = close[i] >= stop or close[i] >= ma[i]
                if exit_now or age >= params.max_hold_bars:
                    position, age = 0, 0
            elif long_entry[i] and close[i] > ma[i]:
                position, age = 1, 0
            elif short_entry[i] and close[i] < ma[i]:
                position, age = -1, 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = EnvelopeParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        close = 100 + direction * np.array([0, 2, 3, -1, -2, -2], dtype=float)
        data = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1000.0
        }, index=index)
        ind = pd.DataFrame({
            "atr": [1.0, 1.0, 5.0, 5.0, 1.0, 1.0],
            "ma200": 100 - direction * 10,
            "floor_strength": float(direction == 1),
            "ceiling_strength": float(direction == -1),
            "floor_rank": 0.9 if direction == 1 else 0.1,
            "ceiling_rank": 0.9 if direction == -1 else 0.1
        }, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        assert result["size"].gt(0).all()
        limited = EnvelopeParams(trail_atr=2.0, max_hold_bars=2)
        result = strategy.generate_signals(data, ind, None, limited).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
