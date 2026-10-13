from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankDisagreementParams:
    rank_window: int = 20
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RankDisagreementParams]):
    strategy_id = "gen_local-w2_1791331964"

    @classmethod
    def params_type(cls):
        return RankDisagreementParams

    @staticmethod
    def warmup_bars(params: RankDisagreementParams) -> int:
        return max(params.rank_window, 15)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RankDisagreementParams) -> pd.DataFrame:
        if params.rank_window < 5 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("rank_window must be >= 5 and stop_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = ((data["close"] - data["low"]) / out["daily_range"]).clip(0.0, 1.0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["price_rank"] = data["close"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["location_rank"] = out["close_location"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["entry"] = (
            out["price_rank"].le(0.25)
            & out["location_rank"].ge(0.80)
            & data["close"].gt(data["open"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RankDisagreementParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                held = 0
                high_water = np.nan
                stop = -np.inf
                continue
            if active:
                held += 1
                high_water = close if np.isnan(high_water) else max(high_water, close)
                stop = max(stop, high_water - params.stop_atr * atr)
                if close < stop or held >= 4:
                    active = False
                    held = 0
                    high_water = np.nan
                    stop = -np.inf
                else:
                    signals[i] = 1
            elif entries[i]:
                active = True
                held = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RankDisagreementParams()
    index = pd.date_range("2020-01-01", periods=6)
    derived = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False]}, index=index)
    for closes, expected in [
        ([10, 12, 11, 9, 10, 10], [0, 1, 1, 1, 0, 0]),
        ([10, 11, 12, 13, 14, 15], [0, 1, 1, 1, 1, 0]),
    ]:
        data = pd.DataFrame({"close": closes}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params)
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()
