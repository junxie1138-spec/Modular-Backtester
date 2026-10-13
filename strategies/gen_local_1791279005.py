from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    wave_bars: int = 3
    variance_window: int = 30
    atr_bars: int = 14
    volatility_ratio: float = 1.20
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if self.wave_bars < 2 or self.variance_window < 3 or self.atr_bars < 1:
            raise ValueError("Invalid indicator windows")
        if not np.isfinite(self.volatility_ratio) or self.volatility_ratio <= 1:
            raise ValueError("volatility_ratio must exceed one")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or self.hold_bars < 1:
            raise ValueError("Invalid exit parameters")


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1791279005"

    @classmethod
    def params_type(cls):
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return int(max(200, params.variance_window + params.wave_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["log_close"] = np.log(close.where(close > 0))
        out["daily_return"] = out["log_close"].diff()
        out["wave_return"] = out["log_close"].diff(params.wave_bars)
        out["daily_std"] = out["daily_return"].rolling(
            params.variance_window, min_periods=params.variance_window).std(ddof=0)
        out["wave_std"] = out["wave_return"].rolling(
            params.variance_window, min_periods=params.variance_window).std(ddof=0)
        out["volatility_ratio"] = out["wave_std"] / (
            np.sqrt(params.wave_bars) * out["daily_std"].replace(0, np.nan))
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        out["displacement"] = close.diff(params.wave_bars)
        out["release"] = (
            out["volatility_ratio"].gt(params.volatility_ratio)
            & out["volatility_ratio"].shift(1).le(params.volatility_ratio))
        long_entry = out["release"] & close.gt(out["ma200"]) & out["displacement"].gt(out["atr"])
        short_entry = out["release"] & close.lt(out["ma200"]) & out["displacement"].lt(-out["atr"])
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        valid = np.isfinite(close) & np.isfinite(atr) & np.isfinite(regime) & (atr > 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            if not valid[i]:
                position = 0
            elif position:
                age += 1
                # Start the close watermark on the first bar after the entry decision.
                if position == 1:
                    extreme = close[i] if age == 1 else max(extreme, close[i])
                    candidate = extreme - params.stop_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close[i] <= stop
                else:
                    extreme = close[i] if age == 1 else min(extreme, close[i])
                    candidate = extreme + params.stop_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close[i] >= stop
                wrong_regime = position * (close[i] - regime[i]) <= 0
                if stopped or wrong_regime or age >= params.hold_bars:
                    position = 0
            elif entries[i] and entries[i] * (close[i] - regime[i]) > 0:
                position = int(entries[i])
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = MomentumParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    index = pd.date_range("2020-01-01", periods=8)
    base = np.array([100, 101, 105, 104, 103, 100, 99, 98], dtype=float)
    for direction in (1, -1):
        close = base if direction == 1 else 200 - base
        data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": [1, 1, 1, 10, 10, 1, 1, 1],
                            "ma200": 90 if direction == 1 else 110,
                            "entry": [direction, 0, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        assert (result["size"] > 0).all()
        expired = strategy.generate_signals(data, ind, ctx, MomentumParams(hold_bars=2)).data
        assert expired["signal"].tolist() == [0, direction, direction, 0, 0, 0, 0, 0]
        missing = ind.copy()
        missing["atr"] = np.nan
        assert strategy.generate_signals(data, missing, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(MomentumParams(variance_window=250)) >= 254


if __name__ == "__main__":
    _self_check()
