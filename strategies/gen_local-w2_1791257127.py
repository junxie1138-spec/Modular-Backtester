from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    correlation_window: int = 40
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791257127"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return max(int(params.correlation_window) + 68, 15)

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.correlation_window < 3 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("correlation_window must be >= 3 and stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(5)
        out["weekly_return"] = close.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        window = int(params.correlation_window)
        out["echo"] = out["return"].rolling(window, min_periods=window).corr(out["lagged_return"]).replace([np.inf, -np.inf], np.nan)
        out["echo_rank"] = out["echo"].rolling(63, min_periods=63).rank(pct=True)
        out["previous_rank"] = out["echo_rank"].shift(1)
        continuation = (out["echo_rank"] >= 0.8) & (out["previous_rank"] < 0.8) & (out["echo"] > 0)
        reversal = (out["echo_rank"] <= 0.2) & (out["previous_rank"] > 0.2) & (out["echo"] < 0)
        direction = np.sign(out["weekly_return"].fillna(0))
        valid = np.isfinite(out[["echo", "echo_rank", "previous_rank", "weekly_return", "atr"]]).all(axis=1) & (out["atr"] > 0)
        out["entry"] = np.where(valid & continuation, direction, np.where(valid & reversal, -direction, 0)).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        age = 0
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                if age == 0:
                    # The previous close requested entry; this bar supplies its open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    stop = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                stopped = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < stop) or
                    (position == -1 and closes[i] > stop)
                )
                if stopped or age >= 8:
                    position = 0
                    age = 0
                signals[i] = position
                continue
            if entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_atr = atr[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 107.0, "low": 93.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 94.0
    data.loc[index[6], "close"] = 106.0
    derived = pd.DataFrame({"atr": 2.0, "entry": 0}, index=index)
    derived.loc[index[0], "entry"] = 1
    derived.loc[index[4], "entry"] = -1
    derived.loc[index[8], "entry"] = 1
    result = GeneratedStrategy.generate_signals(None, data, derived, None, EchoParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
