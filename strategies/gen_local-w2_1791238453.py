from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 20
    atr_period: int = 14
    volume_multiple: float = 1.1
    min_separation: float = 0.15
    stop_atr: float = 1.75
    hold_bars: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791238453"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.lookback, params.atr_period) + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not all(np.isfinite(x) and x > 0 for x in (
            params.volume_multiple, params.stop_atr, params.size
        )) or not np.isfinite(params.min_separation) or params.min_separation < 0:
            raise ValueError("Invalid threshold or sizing parameter.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["move"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["up_volume"] = data["volume"].clip(lower=0).where(out["move"] > 0, 0.0)
        out["down_volume"] = data["volume"].clip(lower=0).where(out["move"] < 0, 0.0)
        for side in ("up", "down"):
            volume = out[side + "_volume"]
            out[side + "_turnover"] = close * volume
            out[side + "_volume_sum"] = volume.rolling(params.lookback).sum().shift(1)
            out[side + "_turnover_sum"] = out[side + "_turnover"].rolling(params.lookback).sum().shift(1)
            out[side + "_anchor"] = (
                out[side + "_turnover_sum"]
                / out[side + "_volume_sum"].replace(0.0, np.nan)
            )
        out["volume_baseline"] = data["volume"].rolling(params.lookback).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"].replace(0.0, np.nan)
        out["anchor_polarity"] = (
            (out["down_anchor"] - out["up_anchor"]) / out["atr"].replace(0.0, np.nan)
        )
        backed = out["relative_volume"] >= params.volume_multiple
        long = backed & (out["move"] > 0) & (close > out["down_anchor"]) & (
            out["anchor_polarity"] > params.min_separation
        )
        short = backed & (out["move"] < 0) & (close < out["up_anchor"]) & (
            out["anchor_polarity"] < -params.min_separation
        )
        out["candidate"] = np.select([long, short], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                # The next open becomes observable only on the actual entry bar.
                if i == entry_bar:
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if not np.isfinite(closes[i]) or not np.isfinite(stop) or stopped or expired:
                    position = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                position = int(entries[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                stop = np.nan
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(stop_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": [100.0, 100.0, 98.0, 100.0, 100.0, 102.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "atr": 1.0, "confirmed": [0, 1, 0, 0, -1, 0, 0]
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    ind["confirmed"] = [1, 0, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [
        0, 1, 1, 1, 1, 0, 0
    ]
    actual = strategy.indicators(data, params)
    assert actual["confirmed"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
