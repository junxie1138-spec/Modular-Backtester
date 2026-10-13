from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRangeParams:
    seasonal_window: int = 126
    fast_atr_window: int = 5
    slow_atr_window: int = 20
    compression_threshold: float = 1.0
    downside_share_threshold: float = 0.48
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[SeasonalRangeParams]):
    strategy_id = "gen_local_1790967302"

    @classmethod
    def params_type(cls) -> type[SeasonalRangeParams]:
        return SeasonalRangeParams

    @staticmethod
    def warmup_bars(params: SeasonalRangeParams) -> int:
        return int(max(params.seasonal_window + 2,
                       params.fast_atr_window + 2,
                       params.slow_atr_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalRangeParams) -> pd.DataFrame:
        if min(params.seasonal_window, params.fast_atr_window,
               params.slow_atr_window, params.max_hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if not np.isfinite(params.compression_threshold) or params.compression_threshold <= 0:
            raise ValueError("compression_threshold must be finite and positive")
        if not np.isfinite(params.downside_share_threshold) or not 0 < params.downside_share_threshold < 1:
            raise ValueError("downside_share_threshold must lie between zero and one")

        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["true_range"] = true_range
        out["atr_fast"] = true_range.rolling(params.fast_atr_window).mean()
        out["atr_slow"] = true_range.rolling(params.slow_atr_window).mean()
        out["compression"] = out["atr_fast"] / out["atr_slow"].replace(0.0, np.nan)
        out["close_change"] = data["close"].diff()
        out["range_energy"] = true_range.pow(2)
        out["downside_energy"] = out["range_energy"].where(out["close_change"] < 0, 0.0).where(true_range.notna())
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekday"] = weekday
        out["seasonal_downside_share"] = np.nan
        for day in range(5):
            mask = weekday.eq(day)
            total = out["range_energy"].where(mask, 0.0).where(true_range.notna())
            downside = out["downside_energy"].where(mask, 0.0).where(true_range.notna())
            total = total.shift(1).rolling(params.seasonal_window).sum()
            downside = downside.shift(1).rolling(params.seasonal_window).sum()
            share = downside / total.replace(0.0, np.nan)
            out.loc[mask, "seasonal_downside_share"] = share.loc[mask]

        out["qualified"] = (
            out["seasonal_downside_share"].lt(params.downside_share_threshold)
            & out["compression"].lt(params.compression_threshold)
            & out["close_change"].gt(0.0)
            & out["atr_slow"].gt(0.0)
        ).fillna(False)
        out["entry_ready"] = out["qualified"] & out["qualified"].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalRangeParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr_slow"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        highest_close = np.nan
        stop = np.nan
        held = 0

        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                held += 1
                if valid:
                    # Initialize on the entry-fill bar, not the preceding decision bar.
                    highest_close = close[i] if np.isnan(highest_close) else max(highest_close, close[i])
                    candidate = highest_close - params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                if not valid or close[i] <= stop or held >= params.max_hold_bars:
                    active = False
                    highest_close = np.nan
                    stop = np.nan
                    held = 0
                else:
                    signals[i] = 1
            elif valid and ready[i]:
                active = True
                highest_close = np.nan
                stop = np.nan
                held = 0
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-06", periods=5, freq="B")
    sample = pd.DataFrame({"close": [100.0, 101.0, 102.0, 98.0, 99.0]}, index=index)
    derived = pd.DataFrame({"atr_slow": 1.0, "entry_ready": True}, index=index)
    result = GeneratedStrategy.generate_signals(sample, derived, None, SeasonalRangeParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
