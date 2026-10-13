from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    compression_window: int = 4
    baseline_window: int = 40
    compression_ratio: float = 0.35
    atr_period: int = 14
    stop_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790973473"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.baseline_window + params.compression_window + 2,
                       params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if (params.compression_window < 2 or params.baseline_window < 2
                or params.atr_period < 1 or not 3 <= params.hold_days <= 5
                or not 0 < params.compression_ratio < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid compression, ATR, stop, or holding parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        returns = out["return"]
        out["return_floor"] = returns.rolling(params.compression_window).min()
        out["return_ceiling"] = returns.rolling(params.compression_window).max()
        # The reference envelope ends before the entire compression window.
        out["baseline_floor"] = returns.rolling(params.baseline_window).min().shift(
            params.compression_window)
        out["baseline_ceiling"] = returns.rolling(params.baseline_window).max().shift(
            params.compression_window)
        out["return_width"] = out["return_ceiling"] - out["return_floor"]
        out["baseline_width"] = out["baseline_ceiling"] - out["baseline_floor"]
        out["compression"] = out["return_width"] / out["baseline_width"].where(
            out["baseline_width"] > 0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["setup"] = (
            (out["return_floor"] > 0)
            & (out["compression"] <= params.compression_ratio)
            & (out["return_ceiling"] < out["baseline_ceiling"])
            & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        frozen_atr = 0.0
        fixed_stop = np.nan

        for i in range(len(data)):
            if entry_bar >= 0:
                # Observe the entry open only when that bar has arrived.
                if i == entry_bar:
                    fixed_stop = opens[i] - params.stop_atr * frozen_atr
                stopped = (not np.isfinite(fixed_stop)
                           or not np.isfinite(closes[i]) or closes[i] < fixed_stop)
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif (confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                entry_bar = i + 1
                frozen_atr = atr[i]
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 111.0, "low": 99.0,
        "close": [100.0, 100.0, 102.0, 110.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "confirmed": [False, True, False, False, False, False, False, False],
        "atr": [2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    data.loc[index[2], "open"] = 110.0
    data.loc[index[2], "close"] = 108.0
    data.loc[index[3], "close"] = 106.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]

    returns = np.r_[np.tile([0.01, -0.01], 24), np.full(12, 0.001)]
    close = 100.0 * np.cumprod(1.0 + returns)
    index = pd.date_range("2020-01-01", periods=len(close))
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    confirmed = indicators["confirmed"]
    assert confirmed.any()
    assert confirmed.equals(
        indicators["setup"] & indicators["setup"].shift(1, fill_value=False))
    assert not confirmed.iloc[:strategy.warmup_bars(params) - 1].any()
    full = strategy.generate_signals(data, indicators, ctx, params).data
    prefix = data.iloc[:55]
    partial = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:55], partial)


if __name__ == "__main__":
    _self_check()
