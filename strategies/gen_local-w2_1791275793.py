from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    activation_pct: float = 0.004


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791275793"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.ma_window + 3, 15)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.ma_window < 5 or not np.isfinite(params.activation_pct) or params.activation_pct <= 0:
            raise ValueError("ma_window must be >= 5 and activation_pct must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["z_previous"] = out["z"].shift(1)
        out["z_earlier"] = out["z"].shift(2)
        out["ma_rise"] = out["ma"] - out["ma"].shift(3)
        out["prior_high"] = data["high"].rolling(5, min_periods=5).max().shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            out["z"].between(0.5, 1.5)
            & (out["z_previous"] > 0)
            & (out["z_previous"] < out["z_earlier"])
            & (out["z"] > out["z_previous"])
            & (out["ma_rise"] > 0)
            & (close < out["prior_high"])
            & (close > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = stop = np.nan
        armed = False
        bars = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (opening, high, low, close) in enumerate(bars):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The preceding close requested a fill at this open.
                    entry_price = opening
                    stop = entry_price - 0.8 * atr[i - 1]
                    armed = False
                if not np.isfinite([opening, high, low, close, atr[i], stop]).all():
                    entry_bar = -1
                    continue
                # Test the existing stop before raising it: daily OHLC cannot
                # establish whether today's low preceded today's high.
                stopped = low <= stop
                if not stopped:
                    armed = armed or high >= entry_price * (1.0 + params.activation_pct)
                    if armed:
                        stop = max(stop, entry_price, high - 0.8 * atr[i])
                    stopped = close <= stop
                # Exit decisions execute next open, including stop breaches.
                if stopped or i - entry_bar + 1 >= 2:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100, 100, 100.6, 100.2, 100.2],
        "high": [100.2, 101, 100.7, 100.4, 100.4],
        "low": [99.8, 99.9, 99.9, 100, 100],
        "close": [100, 100.6, 100.2, 100.2, 100.2],
        "volume": [1000] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 5, "entry": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", 0)
    signals = strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 99.9
    signals = strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0]
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert not strategy.indicators(flat, GeneratedParams()).entry.any()
