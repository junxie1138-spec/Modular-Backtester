from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRankParams:
    channel_bars: int = 20
    percentile_bars: int = 126
    gap_percentile: float = 0.80
    atr_bars: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapRankParams]):
    strategy_id = "gen_local-w2_1790974996"

    @classmethod
    def params_type(cls):
        return GapRankParams

    def warmup_bars(self, params: GapRankParams) -> int:
        return int(max(params.channel_bars + params.percentile_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapRankParams) -> pd.DataFrame:
        if (min(params.channel_bars, params.percentile_bars, params.atr_bars) < 2
                or not 0.5 < params.gap_percentile < 1
                or not np.isfinite(params.trail_atr)
                or params.trail_atr <= 0 or params.hold_bars < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        history = data["close"].rolling(params.channel_bars, min_periods=params.channel_bars)
        out["floor"] = history.min().shift(1)
        out["ceiling"] = history.max().shift(1)
        out["lower_quartile"] = history.quantile(0.25).shift(1)
        out["median"] = history.quantile(0.50).shift(1)
        out["upper_quartile"] = history.quantile(0.75).shift(1)
        out["width"] = (out["ceiling"] - out["floor"]).replace(0, np.nan)
        out["gap"] = (data["open"] - out["prior_close"]) / out["width"]
        out["gap_threshold"] = out["gap"].abs().shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.gap_percentile)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        ready = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        inside = (data["close"].between(out["floor"], out["ceiling"])
                  & data["open"].between(out["floor"], out["ceiling"]))
        large = out["gap"].abs() > out["gap_threshold"]
        long_entry = (ready & inside & large & (out["gap"] > 0)
                      & (out["prior_close"] <= out["lower_quartile"])
                      & (data["open"] > out["median"])
                      & (data["close"] > out["median"]))
        short_entry = (ready & inside & large & (out["gap"] < 0)
                       & (out["prior_close"] >= out["upper_quartile"])
                       & (data["open"] < out["median"])
                       & (data["close"] < out["median"]))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRankParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered = 0
        extreme = np.nan
        stop = np.nan
        for i, (close, atr, entry) in enumerate(zip(
                data["close"].to_numpy(), indicators["atr"].to_numpy(),
                indicators["entry"].to_numpy())):
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if position:
                if not valid:
                    position = 0
                else:
                    # Entry fills on the next bar; track closes starting on that bar.
                    if position == 1:
                        extreme = close if np.isnan(extreme) else max(extreme, close)
                        candidate = extreme - params.trail_atr * atr
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        hit = close <= stop
                    else:
                        extreme = close if np.isnan(extreme) else min(extreme, close)
                        candidate = extreme + params.trail_atr * atr
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        hit = close >= stop
                    if hit or i - entered >= params.hold_bars:
                        position = 0
            elif valid and entry in (-1, 1):
                position = int(entry)
                entered = i
                extreme = np.nan
                stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapRankParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 103., 104., 103., 98., 98.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 10., 10., 1.],
        "entry": [1, 0, 0, 0, 0, 0],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
