from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 6
    breakout_window: int = 10
    atr_window: int = 14
    volume_window: int = 20
    area_threshold: float = 0.06
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2
    max_hold_bars: int = 2
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791330167"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(
            params.atr_window + params.signature_window,
            params.volume_window + params.signature_window - 1,
            params.breakout_window + 1,
        )

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.signature_window, params.breakout_window, params.atr_window, params.volume_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not (0.0 <= params.area_threshold < 0.5):
            raise ValueError("area_threshold must be in [0, 0.5)")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Stop parameters must be positive")
        if params.max_hold_bars not in (1, 2) or not (0 < params.min_size <= 1):
            raise ValueError("Holding period must be one or two bars and min_size in (0, 1]")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["volume_mean"] = data["volume"].rolling(params.volume_window).mean()
        out["price_step"] = data["close"].diff() / out["atr"].shift(1).replace(0, np.nan)
        candle_range = (data["high"] - data["low"]).replace(0, np.nan)
        out["close_pressure"] = ((2 * data["close"] - data["high"] - data["low"]) / candle_range).fillna(0).clip(-1, 1)
        out["pressure_step"] = out["close_pressure"] * data["volume"] / out["volume_mean"].replace(0, np.nan)
        x = out["price_step"]
        y = out["pressure_step"]
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.signature_window):
            cross = y.shift(lag) * x - x.shift(lag) * y
            area = area + cross.rolling(params.signature_window - lag).sum()
        out["raw_area"] = 0.5 * area
        out["price_variation"] = x.abs().rolling(params.signature_window).sum()
        out["pressure_variation"] = y.abs().rolling(params.signature_window).sum()
        denominator = out["price_variation"] * out["pressure_variation"]
        out["signature_area"] = (out["raw_area"] / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["prior_channel_high"] = data["high"].rolling(params.breakout_window).max().shift(1)
        out["conviction"] = ((out["signature_area"] - params.area_threshold) / (0.5 - params.area_threshold)).clip(0, 1).fillna(0)
        out["entry"] = (
            (data["close"] > out["prior_channel_high"])
            & (out["signature_area"] > params.area_threshold)
            & (out["atr"] > 0)
            & (out["pressure_step"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        active = False
        armed = False
        entry_price = stop = position_size = 0.0
        held = 0
        warmup = self.warmup_bars(params)
        for i in range(count):
            if not active and i > 0 and decisions[i - 1] == 1:
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue
                active = True
                armed = False
                held = 0
                position_size = sizes[i - 1]
                stop = entry_price - params.trail_atr * atr[i - 1]
            if active:
                held += 1
                sizes[i] = position_size
                # Stop breaches request an exit at the next open, matching the signal contract.
                if lows[i] <= stop or held >= params.max_hold_bars:
                    active = False
                    continue
                if highs[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    # Ratchet at the close; today's low never tests a newly raised stop.
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]):
                        stop = max(stop, closes[i] - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif i >= warmup - 1 and entries[i]:
                decisions[i] = 1
                sizes[i] = params.min_size + (1.0 - params.min_size) * conviction[i]
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.1, "volume": 1000.0}, index=index)
    params = SignatureParams(signature_window=2, breakout_window=2, atr_window=2, volume_window=2)
    strategy = GeneratedStrategy()
    features = strategy.indicators(data, params)
    features["entry"] = False
    features.loc[index[4], "entry"] = True
    features["atr"] = 1.0
    features["conviction"] = 0.5
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    assert np.isclose(result.loc[index[5], "size"], 0.625)
    data.loc[index[5], ["high", "close"]] = [101.0, 100.8]
    data.loc[index[6], "low"] = 99.95
    params.max_hold_bars = 2
    result = strategy.generate_signals(data, features, None, params).data
    assert result.loc[index[7], "signal"] == 0
    data.loc[index[5], "low"] = 98.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result.loc[index[6], "signal"] == 0
