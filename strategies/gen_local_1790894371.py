from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_period: int = 20
    atr_period: int = 14
    change_threshold: float = 1.25
    stretch_threshold: float = 0.9
    stop_atr: float = 1.5
    hold_days: int = 4
    size: float = 1.0

    def __post_init__(self):
        if self.ma_period < 2 or self.atr_period < 1 or not 3 <= self.hold_days <= 5:
            raise ValueError("Invalid lookback or holding period")
        values = (self.change_threshold, self.stretch_threshold, self.stop_atr, self.size)
        if not all(np.isfinite(v) and v > 0 for v in values) or self.size > 1:
            raise ValueError("Thresholds and size must be finite and positive; size <= 1")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790894371"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(200, params.ma_period + 12, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_period).mean()
        out["price_std"] = close.rolling(params.ma_period).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["recent_z"] = out["z"].rolling(3).mean()
        out["baseline_z"] = out["z"].shift(3).rolling(8).mean()
        out["baseline_std"] = out["z"].shift(3).rolling(8).std(ddof=0)
        out["change"] = (out["recent_z"] - out["baseline_z"]) / out["baseline_std"].clip(lower=0.25)
        out["previous_change"] = out["change"].shift(1)
        out["regime_ma"] = close.rolling(200).mean()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators[
            ["z", "baseline_z", "change", "previous_change", "regime_ma", "atr"]
        ]).all(axis=1)
        entry = (
            valid
            & (data["close"] > indicators["regime_ma"])
            & (indicators["baseline_z"] <= -params.stretch_threshold)
            & (indicators["z"] < 0)
            & (indicators["change"] >= params.change_threshold)
            & (indicators["previous_change"] < params.change_threshold)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_decision = -1
        frozen_atr = stop = np.nan
        for i in range(len(data)):
            if active:
                # Entry fills today: anchor once to this open and decision-bar ATR.
                if i == entry_decision + 1:
                    stop = opens[i] - params.stop_atr * frozen_atr
                expired = i - entry_decision >= params.hold_days
                stopped = not np.isfinite(stop) or not np.isfinite(closes[i]) or closes[i] < stop
                if expired or stopped:
                    active = False
            elif entry[i] and np.isfinite(closes[i]):
                active = True
                entry_decision = i
                frozen_atr = atr[i]
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 111.0, "low": 98.0,
         "close": [100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"z": -0.5, "baseline_z": -1.0, "change": [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
         "previous_change": [np.nan, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0],
         "regime_ma": 90.0, "atr": 1.0}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    data.loc[index[3], "close"] = 98.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    indicators["regime_ma"] = 200.0
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
