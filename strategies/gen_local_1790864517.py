from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class IntradayBreakoutParams:
    rank_window: int = 40
    path_threshold: float = 0.90
    price_threshold: float = 0.75
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[IntradayBreakoutParams]):
    strategy_id = "gen_local_1790864517"

    @classmethod
    def params_type(cls) -> type[IntradayBreakoutParams]:
        return IntradayBreakoutParams

    def warmup_bars(self, params: IntradayBreakoutParams) -> int:
        return max(params.rank_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: IntradayBreakoutParams) -> pd.DataFrame:
        if params.rank_window < 2 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive; rank_window must exceed one.")
        if not (0.5 < params.path_threshold < 1.0 and 0.5 < params.price_threshold < 1.0):
            raise ValueError("Percentile thresholds must lie strictly between 0.5 and 1.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        valid_prices = (data["open"] > 0) & (close > 0)
        out["intraday_return"] = np.log(close.where(valid_prices) / data["open"].where(valid_prices)).replace([np.inf, -np.inf], np.nan)
        out["intraday_path"] = out["intraday_return"].cumsum()
        out["path_rank"] = out["intraday_path"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["price_rank"] = close.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["ready"] = np.isfinite(out[["intraday_return", "path_rank", "price_rank", "atr"]]).all(axis=1) & (out["atr"] > 0)
        out["long_candidate"] = out["ready"] & (out["path_rank"] >= params.path_threshold) & (out["price_rank"] >= params.price_threshold) & (out["intraday_return"] > 0)
        out["short_candidate"] = out["ready"] & (out["path_rank"] <= 1.0 - params.path_threshold) & (out["price_rank"] <= 1.0 - params.price_threshold) & (out["intraday_return"] < 0)
        out["long_confirmed"] = out["long_candidate"] & out["long_candidate"].shift(1, fill_value=False)
        out["short_confirmed"] = out["short_candidate"] & out["short_candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: IntradayBreakoutParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        longs = indicators["long_confirmed"].to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = 0.0
        low_water = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if position != 0:
                if not ready[i] or not np.isfinite(close):
                    position = 0
                else:
                    if position == 1:
                        high_water = max(high_water, close)
                        stop = max(stop, high_water - params.trail_atr * atr)
                        stopped = close <= stop
                    else:
                        low_water = min(low_water, close)
                        stop = min(stop, low_water + params.trail_atr * atr)
                        stopped = close >= stop
                    if stopped or i - entry_bar >= params.max_hold:
                        position = 0
            elif ready[i] and np.isfinite(close):
                if longs[i]:
                    position = 1
                    high_water = close
                    stop = close - params.trail_atr * atr
                    entry_bar = i
                elif shorts[i]:
                    position = -1
                    low_water = close
                    stop = close + params.trail_atr * atr
                    entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
