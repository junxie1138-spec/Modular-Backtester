from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_lookback: int = 20
    streak_length: int = 4
    compression_ratio: float = 0.85
    pressure_window: int = 5
    pressure_threshold: float = 0.25
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791260097"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.range_lookback + params.streak_length + 1,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if (params.range_lookback < 2 or
                min(params.streak_length, params.pressure_window,
                    params.atr_window, params.hold_bars) < 1 or
                not 0 < params.compression_ratio < 1 or
                not 0 < params.pressure_threshold <= 1 or
                not params.breakeven_pct > 0 or not params.trail_atr > 0):
            raise ValueError("Invalid compression strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["bar_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["normalized_range"] = out["bar_range"] / data["close"].where(data["close"] > 0)
        out["range_reference"] = out["normalized_range"].rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).median().shift(1)
        out["compressed"] = (
            out["normalized_range"] < params.compression_ratio * out["range_reference"]
        ).fillna(False)
        groups = (~out["compressed"]).cumsum()
        out["compression_streak"] = out["compressed"].astype(int).groupby(groups).cumsum()
        out["close_location"] = (
            (2 * data["close"] - data["high"] - data["low"]) /
            out["bar_range"].replace(0, np.nan)
        ).clip(-1, 1).fillna(0.0)
        out["signed_volume"] = out["close_location"] * data["volume"].clip(lower=0)
        out["pressure_volume"] = data["volume"].clip(lower=0).rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure_sum"] = out["signed_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure"] = out["pressure_sum"] / out["pressure_volume"].replace(0, np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["bar_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        ready = (
            (out["compression_streak"] >= params.streak_length) &
            out["range_reference"].notna() &
            np.isfinite(out["pressure"]) & np.isfinite(out["atr"]) & (out["atr"] > 0)
        )
        out["entry"] = np.select([
            ready & (out["pressure"] >= params.pressure_threshold),
            ready & (out["pressure"] <= -params.pressure_threshold)
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        side = pending = 0
        pending_atr = entry_price = stop = 0.0
        entry_bar = 0
        armed = False
        for i in range(len(data)):
            if not (np.isfinite(opens[i]) and opens[i] > 0 and
                    np.isfinite(closes[i]) and closes[i] > 0 and
                    np.isfinite(atrs[i]) and atrs[i] > 0):
                side = pending = 0
                continue
            if pending:
                side = pending
                pending = 0
                entry_price = opens[i]
                entry_bar = i
                armed = False
                stop = side * entry_price - params.trail_atr * pending_atr
            if side:
                # Signed prices make the stop ratchet upward for both directions.
                signed_close = side * closes[i]
                if (signed_close <= stop or
                        i - entry_bar + 1 >= params.hold_bars):
                    side = 0
                else:
                    if signed_close - side * entry_price >= entry_price * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry_price,
                                   signed_close - params.trail_atr * atrs[i])
                    signals[i] = side
                # Exit decisions fill next open; never reverse on the exit bar.
                continue
            if entries[i]:
                pending = int(entries[i])
                pending_atr = atrs[i]
                signals[i] = pending
        # ponytail: stops use closes; intraday execution requires a different engine contract.
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        prices = pd.DataFrame({
            "open": [100, 100, 102, 104, 100, 100],
            "close": [100, 101, 103, 104, 100, 100]
        }, index=index)
        if direction == -1:
            prices = 200 - prices
        ind = pd.DataFrame({"atr": 1.0, "entry": [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(prices, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        params.hold_bars = 2
        result = strategy.generate_signals(prices, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]
        params.hold_bars = 18


if __name__ == "__main__":
    _self_check()
