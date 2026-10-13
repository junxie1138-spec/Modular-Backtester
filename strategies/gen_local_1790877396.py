from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    lookback: int = 8
    compression_window: int = 3
    z_threshold: float = 1.1
    compression_ratio: float = 0.75
    volume_ratio: float = 1.05
    profit_target: float = 0.03
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790877396"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        if not 2 <= params.lookback <= 10:
            raise ValueError("lookback must be between 2 and 10")
        if not 1 <= params.compression_window <= params.lookback:
            raise ValueError("compression_window must not exceed lookback")
        return int(params.lookback)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        limits = [params.z_threshold, params.compression_ratio,
                  params.volume_ratio, params.profit_target]
        if (not np.isfinite(limits).all() or params.z_threshold <= 0
                or not 0 < params.compression_ratio < 1
                or params.volume_ratio < 1 or params.profit_target <= 0
                or params.max_hold_bars < 1):
            raise ValueError("Invalid entry or exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["ma"] = close.rolling(params.lookback).mean()
        out["std"] = close.rolling(params.lookback).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["daily_range"] = (data["high"] - data["low"]).where(
            data["high"] >= data["low"])
        out["range_fast"] = out["daily_range"].rolling(params.compression_window).mean()
        out["range_slow"] = out["daily_range"].rolling(params.lookback).mean()
        out["compression"] = out["range_fast"] / out["range_slow"].replace(0.0, np.nan)
        volume = data["volume"].where(data["volume"] >= 0)
        out["volume_fast"] = volume.rolling(params.compression_window).mean()
        out["volume_slow"] = volume.rolling(params.lookback).mean()
        out["crowding"] = out["volume_fast"] / out["volume_slow"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CompressionParams,
    ) -> SignalFrame:
        ready = ((indicators["compression"] <= params.compression_ratio)
                 & (indicators["crowding"] >= params.volume_ratio))
        entries = np.where(ready & (indicators["z"] <= -params.z_threshold), 1,
                           np.where(ready & (indicators["z"] >= params.z_threshold), -1, 0))
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age, entry_price = 0, 0, np.nan
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at today's open.
                if age == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                # A close-based target or deadline schedules next-open liquidation.
                if age >= params.max_hold_bars or (
                    np.isfinite(gain) and gain >= params.profit_target
                ):
                    position = 0
            elif np.isfinite(closes[i]) and closes[i] > 0 and entries[i]:
                position, age = int(entries[i]), 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams(profit_target=0.02, max_hold_bars=2)
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    for direction, opens, closes in [
        (1, [100, 101, 100, 100, 100], [100, 101, 104, 100, 100]),
        (-1, [100, 99, 100, 100, 100], [100, 99, 96, 100, 100]),
        (1, [100] * 5, [100] * 5),
    ]:
        data = pd.DataFrame({"open": opens, "close": closes}, index=index)
        ind = pd.DataFrame({"z": [-2 * direction, 0, 0, 0, 0],
                            "compression": 0.5, "crowding": 1.2}, index=index)
        # Targets fire before the deadline in the first two cases.
        params.max_hold_bars = 18 if closes[2] != 100 else 2
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 0.0},
                        index=pd.date_range("2020-01-01", periods=24))
    ind = strategy.indicators(flat, params)
    assert strategy.warmup_bars(params) == 8
    assert ind["z"].iloc[:7].isna().all()
    assert strategy.generate_signals(flat, ind, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
