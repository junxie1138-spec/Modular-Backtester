from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InventoryBreakoutParams:
    breakout_bars: int = 12
    volume_bars: int = 20
    atr_bars: int = 14
    opposing_share: float = 0.55
    volume_multiple: float = 1.25
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[InventoryBreakoutParams]):
    strategy_id = "gen_local-w2_1791343918"

    @classmethod
    def params_type(cls) -> type[InventoryBreakoutParams]:
        return InventoryBreakoutParams

    def warmup_bars(self, params: InventoryBreakoutParams) -> int:
        return max(params.breakout_bars + 2, params.volume_bars + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: InventoryBreakoutParams) -> pd.DataFrame:
        if min(params.breakout_bars, params.volume_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not 0.5 < params.opposing_share < 1.0:
            raise ValueError("opposing_share must be between 0.5 and 1.")
        if params.volume_multiple <= 1.0 or params.trail_atr <= 0:
            raise ValueError("Invalid volume or trailing-stop threshold.")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["change"] = close.diff()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["ceiling"] = data["high"].rolling(params.breakout_bars).max().shift(1)
        out["floor"] = data["low"].rolling(params.breakout_bars).min().shift(1)
        volume = data["volume"].clip(lower=0)
        out["prior_volume"] = volume.rolling(params.breakout_bars).sum().shift(1)
        out["down_volume"] = volume.where(out["change"] < 0, 0.0).rolling(
            params.breakout_bars
        ).sum().shift(1)
        out["up_volume"] = volume.where(out["change"] > 0, 0.0).rolling(
            params.breakout_bars
        ).sum().shift(1)
        denominator = out["prior_volume"].replace(0, np.nan)
        out["down_share"] = out["down_volume"] / denominator
        out["up_share"] = out["up_volume"] / denominator
        out["volume_mean"] = volume.rolling(params.volume_bars).mean().shift(1)
        out["volume_ratio"] = volume / out["volume_mean"].replace(0, np.nan)
        atr = out["atr"].where(out["atr"] > 0)
        out["up_escape"] = (close - out["ceiling"]) / atr
        out["down_escape"] = (out["floor"] - close) / atr

        liquid = out["volume_ratio"] >= params.volume_multiple
        long_entry = (
            liquid & (out["up_escape"] > 0) & (close > data["open"])
            & (out["down_share"] >= params.opposing_share)
        )
        short_entry = (
            liquid & (out["down_escape"] > 0) & (close < data["open"])
            & (out["up_share"] >= params.opposing_share)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        out["opposing_inventory"] = out["down_share"].where(
            out["entry"] == 1, out["up_share"]
        )
        out["escape"] = out["up_escape"].where(
            out["entry"] == 1, out["down_escape"]
        )
        out["conviction"] = (
            (out["volume_ratio"] / params.volume_multiple - 1).clip(0, 1)
            * (out["opposing_inventory"] - params.opposing_share).clip(lower=0)
            / (1 - params.opposing_share)
            * out["escape"].clip(0, 1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: InventoryBreakoutParams
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.2}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction"].fillna(0).to_numpy(dtype=float)
        position = 0
        age = 0
        watermark = 0.0
        stop = 0.0
        held_size = 0.2

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    if age == 1:
                        # Entry fills at this bar's open; its ATR was known yesterday.
                        watermark = opens[i] if np.isfinite(opens[i]) else close
                        entry_atr = atrs[i - 1]
                        stop = watermark - position * params.trail_atr * entry_atr
                    if position == 1:
                        watermark = max(watermark, close)
                        stop = max(stop, watermark - params.trail_atr * atr)
                        stopped = close <= stop
                    else:
                        # Symmetric low-water ratchet protects short positions.
                        watermark = min(watermark, close)
                        stop = min(stop, watermark + params.trail_atr * atr)
                        stopped = close >= stop
                    if stopped or age >= params.hold_bars:
                        position = 0
                # An exit occupies this decision bar: no immediate re-entry.
            elif valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                held_size = float(np.clip(0.2 + 0.75 * conviction[i], 0.2, 0.95))
            signals[i] = position
            sizes[i] = held_size if position else 0.2

        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 102, 103, 100, 100, 100, 98, 101, 101], dtype=float)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "entry": [1, 0, 0, 0, -1, 0, 0, 0, 0],
        "conviction": 0.5
    }, index=index)
    strategy = GeneratedStrategy()
    params = InventoryBreakoutParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.575)
    indicators["entry"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    flat = data.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = 100.0
    result = strategy.generate_signals(flat, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
