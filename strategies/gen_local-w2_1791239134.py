from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    corridor_bars: int = 20
    volume_bars: int = 20
    atr_bars: int = 14
    volume_multiple: float = 1.3
    headroom: float = 0.15
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791239134"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if min(params.corridor_bars, params.volume_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not (params.volume_multiple > 0 and params.stop_atr > 0 and 0 <= params.headroom < 0.5):
            raise ValueError("Invalid volume, stop, or headroom parameter.")
        return int(max(params.corridor_bars, params.volume_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        window = close.rolling(params.corridor_bars, min_periods=params.corridor_bars)
        out["q25"] = window.quantile(0.25).shift(1)
        out["q75"] = window.quantile(0.75).shift(1)
        out["floor"] = data["low"].rolling(params.corridor_bars).min().shift(1)
        out["ceiling"] = data["high"].rolling(params.corridor_bars).max().shift(1)
        width = (out["ceiling"] - out["floor"]).replace(0.0, np.nan)
        out["position"] = (close - out["floor"]) / width
        out["volume_baseline"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["change"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["volume_confirmed"] = (
            (out["volume_baseline"] > 0)
            & (data["volume"] >= params.volume_multiple * out["volume_baseline"])
        )
        out["relative_long"] = (
            (out["previous_close"] <= out["q75"]) & (close > out["q75"])
            & out["position"].between(0.5, 1.0 - params.headroom)
        )
        out["relative_short"] = (
            (out["previous_close"] >= out["q25"]) & (close < out["q25"])
            & out["position"].between(params.headroom, 0.5)
        )
        ready = np.isfinite(out[["q25", "q75", "position", "volume_baseline", "change", "atr"]]).all(axis=1)
        ready &= out["atr"] > 0
        long_entry = ready & out["relative_long"] & out["volume_confirmed"] & (out["change"] > 0)
        short_entry = ready & out["relative_short"] & out["volume_confirmed"] & (out["change"] < 0)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        desired = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = 0
        stop = 0.0
        for i in range(len(data)):
            # Previous close's order fills at today's open; freeze the entry ATR.
            if side == 0 and i > 0 and desired[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                side = int(desired[i - 1])
                entry_bar = i
                stop = opens[i] - side * params.stop_atr * atr[i - 1]
            if side != 0:
                stopped = not np.isfinite(closes[i]) or side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                desired[i] = side
            elif candidates[i] != 0 and np.isfinite(closes[i]):
                desired[i] = candidates[i]
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 100.0,
    }, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "candidate": [1, 0, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    # Entry gaps upward: the stop must use the fill open, not the decision close.
    data.loc[index[1], "open"] = 105.0
    data.loc[index[1], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    indicators["candidate"] = [-1, 0, 0, 0, 0, 0, 0, 0]
    data.loc[index[1], "open"] = 100.0
    data.loc[index[1], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
