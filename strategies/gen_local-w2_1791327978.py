from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionEscapeParams:
    channel_bars: int = 12
    volume_bars: int = 20
    volume_multiple: float = 1.5
    minimum_body_fraction: float = 0.55
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AuctionEscapeParams]):
    strategy_id = "gen_local-w2_1791327978"

    @classmethod
    def params_type(cls) -> type[AuctionEscapeParams]:
        return AuctionEscapeParams

    def warmup_bars(self, params: AuctionEscapeParams) -> int:
        return int(max(params.channel_bars, params.volume_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: AuctionEscapeParams) -> pd.DataFrame:
        if params.channel_bars < 1 or params.volume_bars < 1:
            raise ValueError("Lookbacks must be positive.")
        if params.volume_multiple <= 0 or params.profit_target <= 0:
            raise ValueError("Volume multiple and profit target must be positive.")
        if not 0 <= params.minimum_body_fraction <= 1 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Invalid body fraction or holding horizon.")
        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.channel_bars).min().shift(1)
        out["volume_mean"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["volume_mean"].where(out["volume_mean"] > 0)
        out["gap"] = data["open"] - data["close"].shift(1)
        out["body"] = data["close"] - data["open"]
        out["range"] = data["high"] - data["low"]
        valid_range = out["range"].where(out["range"] > 0)
        out["body_fraction"] = out["body"].abs() / valid_range
        out["close_location"] = (data["close"] - data["low"]) / valid_range
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionEscapeParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        volume_confirmed = indicators["volume_ratio"] >= params.volume_multiple
        expanded_body = (indicators["body"].abs() >= indicators["gap"].abs()) & (indicators["body_fraction"] >= params.minimum_body_fraction)
        bullish_auction = expanded_body & (indicators["gap"] > 0) & (indicators["body"] > 0) & (indicators["close_location"] >= 0.8)
        bearish_auction = expanded_body & (indicators["gap"] < 0) & (indicators["body"] < 0) & (indicators["close_location"] <= 0.2)
        long_entry = ready & volume_confirmed & (data["close"] > indicators["channel_high"]) & bullish_auction
        short_entry = ready & volume_confirmed & (data["close"] < indicators["channel_low"]) & bearish_auction
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if direction != 0:
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        direction = 0
                        entry_price = np.nan
                        continue
                age += 1
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or age >= params.hold_bars:
                    direction = 0
                    entry_price = np.nan
                    age = 0
                signals[i] = direction
            elif entries[i] != 0:
                direction = int(entries[i])
                entry_price = np.nan
                age = 0
                signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame({
        "open": [10, 10, 10, 10.5, 12, 12.5, 12.5, 12.5],
        "high": [10.1, 10.1, 10.1, 12.1, 12.6, 12.6, 12.6, 12.6],
        "low": [9.9, 9.9, 9.9, 10.4, 11.9, 12.4, 12.4, 12.4],
        "close": [10, 10, 10, 12, 12.5, 12.5, 12.5, 12.5],
        "volume": [100, 100, 100, 200, 100, 100, 100, 100],
    }, index=pd.date_range("2020-01-01", periods=8))
    strategy = GeneratedStrategy()
    p = AuctionEscapeParams(channel_bars=2, volume_bars=2, profit_target=0.5, hold_bars=3)
    for side in (1, -1):
        bars = sample.copy()
        if side == -1:
            bars["open"] = 30 - sample["open"]
            bars["close"] = 30 - sample["close"]
            bars["high"] = 30 - sample["low"]
            bars["low"] = 30 - sample["high"]
        result = strategy.generate_signals(bars, strategy.indicators(bars, p), None, p)
        assert result.data["signal"].tolist() == [0, 0, 0, 0, side, side, side, 0]
        p.profit_target = 0.01
        result = strategy.generate_signals(bars, strategy.indicators(bars, p), None, p)
        assert result.data["signal"].tolist() == [0, 0, 0, 0, side, 0, 0, 0]
        p.profit_target = 0.5
