from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    box_bars: int = 4
    baseline_bars: int = 20
    atr_bars: int = 14
    compression_multiple: float = 1.8
    impulse_threshold: float = 0.8
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790863738"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.box_bars + 1, params.baseline_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.box_bars, params.baseline_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if params.compression_multiple <= 0 or params.impulse_threshold <= 0:
            raise ValueError("Entry thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["daily_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["baseline_range"] = out["true_range"].rolling(params.baseline_bars).mean().shift(1)
        out["box_high"] = data["high"].rolling(params.box_bars).max().shift(1)
        out["box_low"] = data["low"].rolling(params.box_bars).min().shift(1)
        out["box_width"] = out["box_high"] - out["box_low"]
        out["box_midpoint"] = (out["box_high"] + out["box_low"]) / 2.0
        out["baseline_volume"] = data["volume"].rolling(params.baseline_bars).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["baseline_volume"].where(out["baseline_volume"] > 0)
        out["body_fraction"] = (data["close"] - data["open"]) / out["daily_range"].where(out["daily_range"] > 0)
        out["volume_impulse"] = out["body_fraction"] * out["relative_volume"]
        out["close_change"] = data["close"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["compressed_box"] = (
            (out["box_width"] > 0)
            & (out["baseline_range"] > 0)
            & (out["box_width"] <= params.compression_multiple * out["baseline_range"])
            & (data["close"] > out["box_midpoint"])
            & (data["close"] <= out["box_high"])
        )
        out["buying_pressure"] = (
            (out["volume_impulse"] >= params.impulse_threshold)
            & (out["close_change"] > 0)
        )
        out["entry"] = (out["compressed_box"] & out["buying_pressure"]).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.max_hold < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("Holding period and ATR stop multiple must be positive")
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entered = -1
        high_water = stop = -np.inf
        for i, (close, atr, entry) in enumerate(zip(closes, atrs, entries)):
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if entered >= 0:
                if not valid:
                    entered = -1
                    continue
                # First tracked close belongs to the next-bar entry session.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or i - entered >= params.max_hold:
                    entered = -1
                    continue
                raw[i] = 1
            elif entry and valid:
                entered = i
                high_water = stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    close = np.array([100.0, 102.0, 104.0, 102.4, 102.0, 102.0])
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    ind = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0], "entry": [True, False, False, False, False, False]}, index=index)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    ind["atr"] = 1.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    ind["atr"] = np.nan
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].eq(0).all()
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
