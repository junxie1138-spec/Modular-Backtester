from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WhipsawParams:
    gap_atr: float = 0.20
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[WhipsawParams]):
    strategy_id = "gen_local_1791241207"

    @classmethod
    def params_type(cls) -> type[WhipsawParams]:
        return WhipsawParams

    @staticmethod
    def warmup_bars(params: WhipsawParams) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: WhipsawParams) -> pd.DataFrame:
        if params.gap_atr <= 0 or params.stop_atr <= 0:
            raise ValueError("Parameters must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - previous_close
        out["normalized_gap"] = out["gap"].div(out["previous_atr"].where(out["previous_atr"] > 0))
        out["previous_normalized_gap"] = out["normalized_gap"].shift(1)
        out["prior_peak"] = previous_close.rolling(20, min_periods=20).max()
        out["gap_midpoint"] = (data["open"] + previous_close) / 2.0
        out["previous_failed_gap"] = previous_close < data["close"].shift(2)
        out["ready"] = np.isfinite(out[["atr", "normalized_gap", "previous_normalized_gap", "prior_peak"]]).all(axis=1) & (out["atr"] > 0)
        out["entry"] = (
            out["ready"]
            & (out["previous_normalized_gap"] >= params.gap_atr)
            & out["previous_failed_gap"]
            & (out["normalized_gap"] <= -params.gap_atr)
            & (data["close"] > out["gap_midpoint"])
            & (data["close"] < out["prior_peak"] * 0.99)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WhipsawParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        entry_atr = 0.0
        stop_price = 0.0
        held_bars = 0
        for i in range(len(data)):
            if pending:
                pending = False
                active = np.isfinite(opens[i]) and opens[i] > 0
                if active:
                    stop_price = opens[i] - params.stop_atr * entry_atr
                    held_bars = 0
            if active:
                held_bars += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or held_bars >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = True
                entry_atr = atrs[i]
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4, "volume": [1000.0] * 4}, index=index)
    features = pd.DataFrame({"atr": [1.0] * 4, "entry": [True, False, False, False]}, index=index)
    normal = GeneratedStrategy.generate_signals(prices, features, None, WhipsawParams()).data
    assert normal["signal"].tolist() == [0, 1, 1, 0]
    prices.loc[index[1], "open"] = 103.0
    stopped = GeneratedStrategy.generate_signals(prices, features, None, WhipsawParams()).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0]
