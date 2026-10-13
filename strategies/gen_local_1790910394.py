from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionShiftParams:
    rank_window: int = 40
    prior_window: int = 5
    atr_window: int = 14
    tail_percentile: float = 0.80
    stop_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[AuctionShiftParams]):
    strategy_id = "gen_local_1790910394"

    @classmethod
    def params_type(cls):
        return AuctionShiftParams

    def warmup_bars(self, params: AuctionShiftParams) -> int:
        return max(params.rank_window + 1, params.prior_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: AuctionShiftParams) -> pd.DataFrame:
        if min(params.rank_window, params.prior_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.5 < params.tail_percentile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("Invalid percentile or stop distance.")
        out = pd.DataFrame(index=data.index)
        out["range"] = data["high"] - data["low"]
        out["close_location"] = ((data["close"] - data["low"]) / out["range"].where(out["range"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["location_rank"] = out["close_location"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["prior_location"] = out["close_location"].rolling(params.prior_window, min_periods=params.prior_window).mean().shift(2)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        long_day = (out["location_rank"] >= params.tail_percentile) & (out["close_location"] >= 0.65)
        short_day = (out["location_rank"] <= 1.0 - params.tail_percentile) & (out["close_location"] <= 0.35)
        out["tail_direction"] = np.select([long_day, short_day], [1, -1], default=0).astype(int)
        two_long = (out["tail_direction"] == 1) & (out["tail_direction"].shift(1) == 1) & (out["prior_location"] <= 0.5)
        two_short = (out["tail_direction"] == -1) & (out["tail_direction"].shift(1) == -1) & (out["prior_location"] >= 0.5)
        out["confirmed"] = np.select([two_long, two_short], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionShiftParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Entry fills at this open; freeze the previous close's ATR.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = not np.isfinite(stop) or not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0
                if stopped or i - entry_bar + 1 >= 2:
                    position = 0
            elif entries[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_atr = atr[i]
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AuctionShiftParams()
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction, fill, first_close, expected in [
        (1, 100.0, 100.0, [0, 1, 1, 0]),
        (1, 110.0, 108.0, [0, 1, 0, 0]),
        (-1, 90.0, 92.0, [0, -1, 0, 0]),
    ]:
        data = pd.DataFrame({"open": [100.0, fill, fill, fill], "close": [100.0, first_close, fill, fill]}, index=index)
        derived = pd.DataFrame({"atr": 1.0, "confirmed": [direction, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
    location = np.array([0.5, 0.4, 0.3, 0.2, 0.8, 0.9])
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 99.0 + 2.0 * location, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=6))
    derived = strategy.indicators(data, AuctionShiftParams(rank_window=3, prior_window=2, atr_window=2))
    assert derived["confirmed"].iloc[-2:].tolist() == [0, 1]
    assert derived["confirmed"].iloc[:3].eq(0).all()


if __name__ == "__main__":
    _self_check()
