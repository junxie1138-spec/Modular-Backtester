from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreakoutParams:
    channel_bars: int = 20
    path_bars: int = 6
    atr_bars: int = 14
    area_threshold: float = 0.12
    stop_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if self.channel_bars < 2 or self.path_bars < 2 or self.atr_bars < 1:
            raise ValueError("Lookbacks must be positive; channel and path require at least two bars.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not np.isfinite(self.area_threshold) or self.area_threshold < 0:
            raise ValueError("area_threshold must be finite and nonnegative.")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[SignatureBreakoutParams]):
    strategy_id = "gen_local-w2_1790981365"

    @classmethod
    def params_type(cls):
        return SignatureBreakoutParams

    def warmup_bars(self, params: SignatureBreakoutParams) -> int:
        return int(max(params.channel_bars + 2, params.path_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureBreakoutParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        atr = atr.where(np.isfinite(atr) & (atr > 0))
        length = params.path_bars
        # Antisymmetric second-order signature of (normalized time, price/ATR).
        # Positive area means gains arrived late; negative area means losses arrived late.
        area = (
            0.5 * (length - 1) * (close + close.shift(length))
            - close.shift(1).rolling(length - 1, min_periods=length - 1).sum()
        ) / (length * atr)
        area = area.replace([np.inf, -np.inf], np.nan)
        # Both confirmation closes must clear the same channel, frozen before either bar.
        upper = data["high"].rolling(params.channel_bars).max().shift(2)
        lower = data["low"].rolling(params.channel_bars).min().shift(2)
        long_confirmed = (
            (close > upper) & (previous_close > upper)
            & (area > params.area_threshold)
            & (area.shift(1) > params.area_threshold)
        ).fillna(False)
        short_confirmed = (
            (close < lower) & (previous_close < lower)
            & (area < -params.area_threshold)
            & (area.shift(1) < -params.area_threshold)
        ).fillna(False)
        long_trigger = long_confirmed & ~long_confirmed.shift(1, fill_value=False)
        short_trigger = short_confirmed & ~short_confirmed.shift(1, fill_value=False)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[long_trigger & atr.notna()] = 1
        entry.loc[short_trigger & atr.notna()] = -1
        return pd.DataFrame(
            {"true_range": true_range, "atr": atr, "signature_area": area,
             "upper": upper, "lower": lower,
             "long_confirmed": long_confirmed, "short_confirmed": short_confirmed,
             "entry": entry},
            index=data.index,
        )

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureBreakoutParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(count):
            if position:
                if i == entry_bar:
                    # Today's open is known now; yesterday's decision did not use it.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                    else:
                        stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = (
                    not np.isfinite(closes[i])
                    or (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and i + 1 < count:
                position = int(entries[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0}, index=index,
    )
    data.iloc[2, data.columns.get_loc("close")] = 101.0
    data.iloc[3, data.columns.get_loc("close")] = 97.0
    data.iloc[6, data.columns.get_loc("close")] = 99.0
    data.iloc[7, data.columns.get_loc("close")] = 103.0
    derived = pd.DataFrame({"atr": 1.0, "entry": 0}, index=index)
    derived.iloc[[1, 8], derived.columns.get_loc("entry")] = 1
    derived.iloc[5, derived.columns.get_loc("entry")] = -1
    # Changing later ATR must not widen either entry's fixed stop.
    derived.iloc[[3, 7], derived.columns.get_loc("atr")] = 100.0
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, SignatureBreakoutParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, SignatureBreakoutParams())
    assert (flat["entry"] == 0).all()
    # A convex time-price path has positive area and confirms only after two breakout closes.
    accelerating = pd.DataFrame(
        {"close": np.arange(40, dtype=float) ** 2 + 100.0}, index=pd.date_range("2021-01-01", periods=40),
    )
    accelerating["open"] = accelerating["close"]
    accelerating["high"] = accelerating["close"] + 0.1
    accelerating["low"] = accelerating["close"] - 0.1
    accelerating["volume"] = 1000.0
    p = SignatureBreakoutParams(channel_bars=3, path_bars=3, atr_bars=2, area_threshold=0.001)
    derived = strategy.indicators(accelerating, p)
    triggers = np.flatnonzero(derived["entry"].to_numpy() == 1)
    assert len(triggers) > 0
    first = int(triggers[0])
    assert first >= strategy.warmup_bars(p) - 1
    assert accelerating["close"].iloc[first - 1] > derived["upper"].iloc[first]
    assert accelerating["close"].iloc[first] > derived["upper"].iloc[first]


if __name__ == "__main__":
    _self_check()
