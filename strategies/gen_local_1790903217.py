from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    history: int = 20
    atr_period: int = 14
    pin_window: int = 3
    pin_ratio: float = 0.65
    depth_atr: float = 1.5
    pressure_threshold: float = 0.10
    stop_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790903217"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.history, params.atr_period, params.pin_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not (
            min(params.history, params.atr_period, params.pin_window) >= 2
            and 0 < params.pin_ratio < 1
            and params.depth_atr > 0
            and 0 <= params.pressure_threshold < 1
            and params.stop_atr > 0
            and params.hold_bars in (1, 2)
        ):
            raise ValueError("Invalid lookbacks, thresholds, stop distance, or holding horizon.")
        out = pd.DataFrame(index=data.index)
        close, high, low = data["close"], data["high"], data["low"]
        out["previous_close"] = close.shift(1)
        out["range"] = high - low
        out["true_range"] = pd.concat(
            [out["range"], (high - out["previous_close"]).abs(),
             (low - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["close_std"] = close.rolling(params.pin_window).std(ddof=0)
        out["low_std"] = low.rolling(params.pin_window).std(ddof=0)
        out["high_std"] = high.rolling(params.pin_window).std(ddof=0)
        scale = out["close_std"].replace(0.0, np.nan)
        out["floor_ratio"] = out["low_std"] / scale
        out["ceiling_ratio"] = out["high_std"] / scale
        out["previous_peak"] = high.shift(1).rolling(params.history).max()
        out["previous_trough"] = low.shift(1).rolling(params.history).min()
        atr = out["atr"].where(out["atr"] > 0)
        out["drawdown"] = (out["previous_peak"] - close) / atr
        out["drawup"] = (close - out["previous_trough"]) / atr
        out["change"] = close.diff()
        out["close_location"] = (
            (2 * close - high - low) / out["range"].where(out["range"] > 0)
        ).fillna(0.0)
        out["signed_volume"] = out["close_location"] * data["volume"]
        out["volume_sum"] = data["volume"].rolling(params.pin_window).sum()
        out["pressure"] = (
            out["signed_volume"].rolling(params.pin_window).sum()
            / out["volume_sum"].where(out["volume_sum"] > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (out["atr"] > 0)
        dispersed = out["close_std"] >= 0.15 * out["atr"]
        # AND: boundary volatility pins the reversal; signed volume agrees.
        out["long_entry"] = (
            ready & dispersed & (out["floor_ratio"] < params.pin_ratio)
            & (out["drawdown"] >= params.depth_atr) & (out["change"] > 0)
            & (out["pressure"] > params.pressure_threshold)
        ).fillna(False)
        out["short_entry"] = (
            ready & dispersed & (out["ceiling_ratio"] < params.pin_ratio)
            & (out["drawup"] >= params.depth_atr) & (out["change"] < 0)
            & (out["pressure"] < -params.pressure_threshold)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price, stop_distance = np.nan, np.nan
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if position:
                if entry_bar == -1:
                    # Observe the fill bar's open only after the prior decision.
                    entry_price, entry_bar = opens[i], i
                stopped = position * (closes[i] - entry_price) < -stop_distance
                expired = i - entry_bar + 1 >= params.hold_bars
                invalid = not np.isfinite(entry_price) or not np.isfinite(closes[i])
                if stopped or expired or invalid:
                    position, entry_bar = 0, -1
                raw[i] = position
                continue
            if np.isfinite(atr[i]) and atr[i] > 0 and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                entry_bar = -1
                # Freeze decision-bar ATR; neither price nor distance trails.
                stop_distance = params.stop_atr * atr[i]
                raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), GeneratedParams()
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for side, first_close, expected in (
        (1, 100.0, [0, 1, 1, 0, 0]),
        (-1, 100.0, [0, -1, -1, 0, 0]),
        (1, 98.0, [0, 1, 0, 0, 0]),
        (-1, 102.0, [0, -1, 0, 0, 0]),
    ):
        data = pd.DataFrame(
            {"open": 100.0, "high": 103.0, "low": 97.0,
             "close": [100.0, first_close, 100.0, 100.0, 100.0],
             "volume": 1000.0}, index=index,
        )
        indicators = pd.DataFrame(
            {"atr": 1.0, "long_entry": [side == 1, False, False, False, False],
             "short_entry": [side == -1, False, False, False, False]}, index=index,
        )
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
