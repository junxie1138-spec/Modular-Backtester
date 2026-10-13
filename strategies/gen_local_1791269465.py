from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetentionParams:
    retention_window: int = 16
    atr_window: int = 14
    vol_window: int = 21
    min_gap_count: int = 3
    entry_threshold: float = 0.35
    reset_threshold: float = 0.10
    hold_bars: int = 18
    atr_multiple: float = 2.5
    target_vol: float = 0.12

    def __post_init__(self):
        if min(self.retention_window, self.atr_window, self.vol_window) < 2:
            raise ValueError("Windows must be at least two.")
        if not 1 <= self.min_gap_count <= self.retention_window or self.hold_bars < 1:
            raise ValueError("Invalid gap count or holding period.")
        if not 0 <= self.reset_threshold < self.entry_threshold <= 1:
            raise ValueError("Require 0 <= reset < entry <= 1.")
        if not np.isfinite(self.atr_multiple) or not np.isfinite(self.target_vol) or min(self.atr_multiple, self.target_vol) <= 0:
            raise ValueError("ATR multiple and target volatility must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapRetentionParams]):
    strategy_id = "gen_local_1791269465"

    @classmethod
    def params_type(cls):
        return GapRetentionParams

    def warmup_bars(self, params):
        return int(max(200, params.retention_window + 1, params.atr_window + 1, params.vol_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["settlement"] = data["close"] - out["prior_close"]
        out["positive_gap"] = out["gap"].clip(lower=0)
        out["negative_gap"] = (-out["gap"]).clip(lower=0)
        out["positive_retained"] = out["settlement"].clip(lower=0).clip(upper=out["positive_gap"])
        out["negative_retained"] = (-out["settlement"]).clip(lower=0).clip(upper=out["negative_gap"])
        for side in ("positive", "negative"):
            out[side + "_total"] = out[side + "_gap"].rolling(params.retention_window).sum()
            out[side + "_survived"] = out[side + "_retained"].rolling(params.retention_window).sum()
            out[side + "_count"] = (out[side + "_gap"] > 0).astype(float).where(out["gap"].notna()).rolling(params.retention_window).sum()
            out[side + "_rate"] = out[side + "_survived"] / out[side + "_total"].replace(0, np.nan)
        out["score"] = (out["positive_rate"] - out["negative_rate"]).where(
            (out["positive_count"] >= params.min_gap_count) & (out["negative_count"] >= params.min_gap_count)
        )
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1).where(out["prior_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["return"] = data["close"].pct_change(fill_method=None)
        out["vol"] = out["return"].rolling(params.vol_window).std(ddof=1) * np.sqrt(252.0)
        out["ma200"] = data["close"].rolling(200).mean()
        out["exposure"] = (params.target_vol / out["vol"].replace(0, np.nan)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        position, entry_bar, stop = 0, -1, np.nan
        frozen_atr, armed = np.nan, True
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        values = indicators[["score", "atr", "vol", "ma200"]].to_numpy(dtype=float)
        for i, (score, atr, vol, ma200) in enumerate(values):
            if position:
                if i == entry_bar:
                    # Anchor to this bar's actual open; ATR was frozen at the previous close.
                    stop = opens[i] - position * params.atr_multiple * frozen_atr
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired or not np.isfinite(stop):
                    position = 0
                signal[i] = position
                continue
            if not np.isfinite([score, atr, vol, ma200, closes[i]]).all() or atr <= 0 or vol <= 0:
                continue
            if abs(score) <= params.reset_threshold:
                armed = True
            direction = 1 if score >= params.entry_threshold and closes[i] > ma200 else (
                -1 if score <= -params.entry_threshold and closes[i] < ma200 else 0
            )
            if armed and direction:
                position, entry_bar, frozen_atr, armed = direction, i + 1, atr, False
            signal[i] = position
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.01).clip(lower=0.01, upper=0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    params = GapRetentionParams(hold_bars=3)
    for direction in (1, -1):
        data = pd.DataFrame({"open": 100.0, "high": 110.0, "low": 90.0, "close": 100.0, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"score": [direction * 0.6] + [0.0] * 6, "atr": 2.0, "vol": 0.2, "ma200": 100.0 - direction, "exposure": 0.6}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0]
        data.loc[index[2], "close"] = 100.0 - direction * 6.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:3])


if __name__ == "__main__":
    _self_check()
