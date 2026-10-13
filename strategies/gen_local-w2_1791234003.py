from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pressure_threshold: float = 0.04
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791234003"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.pressure_threshold) or params.pressure_threshold <= 0:
            raise ValueError("pressure_threshold must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_close"] = np.log(close.where(close > 0))
        out["return"] = out["log_close"].diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        # Eight-session least-squares slope is the latent daily trend estimate.
        out["slope"] = sum(
            (j - 3.5) * out["log_close"].shift(7 - j) for j in range(8)
        ) / 42.0
        out["innovation"] = out["slope"].diff() * close / out["atr"].replace(0, np.nan)
        out["up_share"] = (
            (out["return"] > 0).astype(float).where(out["return"].notna())
            .rolling(12, min_periods=12).mean()
        )
        # Recent directional participation measures remaining adoption capacity.
        out["long_pressure"] = out["innovation"] * (1.0 - out["up_share"])
        out["short_pressure"] = -out["innovation"] * out["up_share"]
        long_entry = (
            (out["slope"] > 0)
            & (out["long_pressure"] > params.pressure_threshold)
            & (out["long_pressure"].shift(1) <= params.pressure_threshold)
        )
        short_entry = (
            (out["slope"] < 0)
            & (out["short_pressure"] > params.pressure_threshold)
            & (out["short_pressure"].shift(1) <= params.pressure_threshold)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position, fill_bar = 0, -1
        entry_atr, stop = 0.0, 0.0
        for i in range(len(data)):
            if position and fill_bar == -1:
                # Entry decided yesterday fills at today's open; ATR stays frozen.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    stop = opens[i] - position * params.stop_atr * entry_atr
                    fill_bar = i
                else:
                    position = 0
            if position:
                stopped = (
                    not np.isfinite(closes[i]) or closes[i] <= 0
                    or (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                if stopped or i - fill_bar + 1 >= 4:
                    position = 0
            elif (
                i + 1 < len(data) and entries[i] != 0
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
            ):
                position = int(entries[i])
                entry_atr = atr[i]
                fill_bar = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "close": [120, 100, 101, 99, 98, 100, 102, 100, 100]},
        index=index
    )
    indicators = pd.DataFrame(
        {"atr": [1, 100, 100, 100, 100, 1, 1, 1, 1],
         "entry": [1, 0, 0, 0, 0, -1, 0, 0, 0]}, index=index
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, 0, 0]
    data.loc[index[2], "close"] = 98.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, -1, 0, 0]
    data["close"] = 100.0
    indicators["entry"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
