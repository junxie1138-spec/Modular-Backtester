from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    overflow_atr: float = 0.6
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local-w2_1791336204"

    @classmethod
    def params_type(cls):
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        return 20

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if not all(np.isfinite(v) and v > 0 for v in (params.overflow_atr, params.stop_atr)):
            raise ValueError("Parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["intraday"] = data["close"] - data["open"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean().replace(0.0, np.nan)
        out["prior_atr"] = out["atr"].shift(1)
        out["up_arrivals"] = out["gap"].clip(lower=0).rolling(5, min_periods=5).sum().shift(1)
        out["down_arrivals"] = (-out["gap"]).clip(lower=0).rolling(5, min_periods=5).sum().shift(1)
        out["up_absorption"] = (-out["intraday"]).clip(lower=0).rolling(5, min_periods=5).sum().shift(1)
        out["down_absorption"] = out["intraday"].clip(lower=0).rolling(5, min_periods=5).sum().shift(1)
        out["up_count"] = (out["gap"] > 0).astype(float).rolling(5, min_periods=5).sum().shift(1)
        out["down_count"] = (out["gap"] < 0).astype(float).rolling(5, min_periods=5).sum().shift(1)
        out["up_overflow"] = out["up_arrivals"] - out["up_absorption"]
        out["down_overflow"] = out["down_arrivals"] - out["down_absorption"]
        out["range_high"] = data["high"].rolling(5, min_periods=5).max().shift(1)
        out["range_low"] = data["low"].rolling(5, min_periods=5).min().shift(1)
        out["long_entry"] = (
            (out["up_count"] >= 3)
            & (out["up_absorption"] >= 0.5 * out["up_arrivals"])
            & (out["up_overflow"] > params.overflow_atr * out["prior_atr"])
            & (out["gap"] > 0.1 * out["prior_atr"])
            & (data["close"] > out["range_high"])
        )
        out["short_entry"] = (
            (out["down_count"] >= 3)
            & (out["down_absorption"] >= 0.5 * out["down_arrivals"])
            & (out["down_overflow"] > params.overflow_atr * out["prior_atr"])
            & (out["gap"] < -0.1 * out["prior_atr"])
            & (data["close"] < out["range_low"])
        )
        out["entry"] = np.select([out["long_entry"], out["short_entry"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if pending:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = pending
                    entry_bar = i
                    # Freeze the stop at the next-open entry using the decision bar's ATR.
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                pending = 0
            if position:
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                # Exit at the following open after four occupied daily bars.
                if stopped or i - entry_bar >= 3:
                    position = 0
                    continue
                signals[i] = position
            elif entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                pending = int(entries[i])
                signals[i] = pending
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 120.0, "low": 80.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[1], ["open", "close"]] = [110.0, 108.0]
    data.loc[index[2], "close"] = 106.0
    data.loc[index[5], ["open", "close"]] = [90.0, 91.0]
    data.loc[index[6], "close"] = 94.0
    indicators = pd.DataFrame({"atr": 2.0, "entry": 0}, index=index)
    indicators.loc[index[[0, 8]], "entry"] = 1
    indicators.loc[index[4], "entry"] = -1
    strategy = GeneratedStrategy()
    params = GapCapacityParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()
    for length in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:length], indicators.iloc[:length], None, params).data
        assert prefix["signal"].equals(result["signal"].iloc[:length])


if __name__ == "__main__":
    _self_check()
