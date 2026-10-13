from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_bars: int = 20
    compression_bars: int = 5
    baseline_bars: int = 40
    compression_ratio: float = 0.70
    entry_z: float = 1.20
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790962678"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.ma_bars + 1, params.compression_bars + 1, params.baseline_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        mean = close.rolling(params.ma_bars).mean()
        std = close.rolling(params.ma_bars).std(ddof=0).replace(0.0, np.nan)
        baseline = tr.rolling(params.baseline_bars).mean().replace(0.0, np.nan)
        out = pd.DataFrame(index=data.index)
        out["mean"] = mean
        out["std"] = std
        out["z"] = (close - mean) / std
        out["tr"] = tr
        out["atr"] = baseline
        out["fast_range"] = tr.rolling(params.compression_bars).mean()
        out["compression"] = out["fast_range"] / baseline
        out["regime"] = close.rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if (params.ma_bars < 2 or params.compression_bars < 1
                or params.baseline_bars < params.compression_bars
                or params.hold_bars < 1 or params.entry_z <= 0
                or params.stop_atr <= 0 or not 0 < params.compression_ratio < 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        z = indicators["z"]
        compressed = indicators["compression"] <= params.compression_ratio
        long_entry = compressed & (data["close"] > indicators["regime"]) & (z.shift(1) < -params.entry_z) & (z >= -params.entry_z) & (z < 0)
        short_entry = compressed & (data["close"] < indicators["regime"]) & (z.shift(1) > params.entry_z) & (z <= params.entry_z) & (z > 0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        valid = np.isfinite(indicators[["z", "atr", "compression", "regime"]].to_numpy(dtype=float)).all(axis=1)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        frozen_atr = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Entry executes at this open; ATR was frozen at the preceding close.
                    stop = opens[i] - position * params.stop_atr * frozen_atr
                stopped = np.isfinite(stop) and position * (close[i] - stop) <= 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid[i] or not np.isfinite(close[i]) or not np.isfinite(stop) or stopped or expired:
                    position = 0
            elif valid[i] and np.isfinite(close[i]) and atr[i] > 0 and entries[i]:
                position = int(entries[i])
                entry_bar = i + 1
                frozen_atr = atr[i]
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=12)
    for side in (1, -1):
        close = np.full(12, 100.0 + side)
        opens = close.copy()
        opens[2] = 100.0 + side * 10.0
        close[2] = 100.0 + side * 11.0
        close[3] = 100.0 + side * 5.0
        data = pd.DataFrame({"open": opens, "high": np.maximum(opens, close) + 1,
                             "low": np.minimum(opens, close) - 1, "close": close,
                             "volume": 1000.0}, index=index)
        features = pd.DataFrame({"z": [-side * 1.8, -side * 0.8] + [0.0] * 10,
                                 "atr": 2.0, "compression": 0.6, "regime": 100.0}, index=index)
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, 0, side, side] + [0] * 8
        data["open"] = 100.0 + side
        data["close"] = 100.0 + side
        result = strategy.generate_signals(data, features, None, params).data
        assert result["signal"].tolist() == [0, 0] + [side] * 8 + [0, 0]
        assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
