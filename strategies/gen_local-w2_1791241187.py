from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 12
    compression_window: int = 3
    percentile_window: int = 60
    atr_window: int = 10
    displacement_percentile: float = 0.80
    compression_percentile: float = 0.25
    breakeven_gain: float = 0.003
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791241187"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ma_window + 1, params.compression_window, params.atr_window + 1) + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["ma_change"] = out["ma"].diff()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["z_threshold"] = out["z"].shift(1).rolling(params.percentile_window).quantile(params.displacement_percentile)
        out["bar_range"] = data["high"] - data["low"]
        out["normalized_range"] = out["bar_range"] / close.replace(0.0, np.nan)
        out["compression"] = out["normalized_range"].rolling(params.compression_window).mean()
        out["compression_threshold"] = out["compression"].shift(1).rolling(params.percentile_window).quantile(params.compression_percentile)
        out["close_location"] = (close - data["low"]) / out["bar_range"].replace(0.0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([out["bar_range"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        eligible = (
            ready
            & (indicators["compression"] <= indicators["compression_threshold"])
            & (indicators["z"] >= indicators["z_threshold"])
            & (indicators["z"] > 0.0)
            & (indicators["ma_change"] > 0.0)
            & (indicators["close_location"] >= 0.5)
            & (indicators["atr"] > 0.0)
        ).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        entry = None
        stop = 0.0
        peak_close = 0.0
        armed = False
        sessions = 0

        for i in range(len(data)):
            # Yesterday's decision fills at today's open.
            if entry is None and i > 0 and desired[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                entry = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                peak_close = entry
                armed = False
                sessions = 0

            if entry is not None:
                sessions += 1
                # Test the previously established stop before raising it.
                # A detected stop breach schedules exit at the next open.
                if not np.isfinite(lows[i]) or not np.isfinite(closes[i]) or lows[i] <= stop or sessions >= 2:
                    entry = None
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]):
                        stop = max(stop, peak_close - params.trail_atr * atr[i])
                if closes[i] <= stop:
                    entry = None
                    continue
                desired[i] = 1
            elif eligible[i]:
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = 100.0 + np.arange(160, dtype=float)
    sample = pd.DataFrame({"open": prices - 0.25, "high": prices + 0.25, "low": prices - 0.75, "close": prices, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=len(prices)))
    strategy = GeneratedStrategy()
    params = Params()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params).data
    prefix = sample.iloc[:120]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert result.iloc[:120].equals(prefix_result)
    assert result["signal"].isin([0, 1]).all() and result["signal"].sum() > 0
    assert result["signal"].rolling(3).sum().max() <= 2
    assert result["signal"].iloc[0] == 0 and (result["size"] > 0.0).all()
