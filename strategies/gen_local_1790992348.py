from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeExhaustionParams:
    memory_window: int = 30
    contraction_threshold: float = 1.0
    stop_atr: float = 2.5
    atr_window: int = 14
    hold_bars: int = 18
    min_correlation: float = 0.25
    min_close_location: float = 0.65


class GeneratedStrategy(BaseStrategy[RangeExhaustionParams]):
    strategy_id = "gen_local_1790992348"

    @classmethod
    def params_type(cls) -> type[RangeExhaustionParams]:
        return RangeExhaustionParams

    def warmup_bars(self, params: RangeExhaustionParams) -> int:
        return int(max(200, params.memory_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeExhaustionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1).where(close.shift(1) > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["relative_range"] = out["daily_range"] / previous_close
        out["range_change"] = out["relative_range"].diff()
        out["lagged_change"] = out["range_change"].shift(1)
        window = params.memory_window
        out["change_mean"] = out["range_change"].rolling(window, min_periods=window).mean().shift(1)
        out["lag_mean"] = out["lagged_change"].rolling(window, min_periods=window).mean().shift(1)
        out["change_std"] = out["range_change"].rolling(window, min_periods=window).std().shift(1)
        out["lag_std"] = out["lagged_change"].rolling(window, min_periods=window).std().shift(1)
        out["correlation"] = out["range_change"].rolling(window, min_periods=window).corr(out["lagged_change"]).shift(1)
        out["slope"] = out["correlation"] * out["change_std"] / out["lag_std"].where(out["lag_std"] > 0)
        out["forecast_change"] = out["change_mean"] + out["slope"] * (out["lagged_change"] - out["lag_mean"])
        out["residual_scale"] = out["change_std"] * np.sqrt((1.0 - out["correlation"].clip(-1.0, 1.0) ** 2).clip(lower=0))
        out["innovation"] = (out["range_change"] - out["forecast_change"]) / out["residual_scale"].where(out["residual_scale"] > 0)
        out["prior_expansion"] = out["range_change"].shift(1).rolling(3, min_periods=3).sum()
        out["close_location"] = (close - data["low"]) / out["daily_range"].where(out["daily_range"] > 0)
        out["true_range"] = pd.concat([
            out["daily_range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeExhaustionParams) -> SignalFrame:
        close = data["close"]
        eligible = (
            (close > indicators["ma200"])
            & (indicators["correlation"] >= params.min_correlation)
            & (indicators["prior_expansion"] > 0)
            & (indicators["forecast_change"] > 0)
            & (indicators["range_change"] < 0)
            & (indicators["innovation"] <= -params.contraction_threshold)
            & (indicators["close_location"] >= params.min_close_location)
            & (close > data["open"])
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = close.to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    # The previous close ordered entry at this bar's open.
                    # Freeze the stop using only ATR known at that decision.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                stopped = not np.isfinite(closes[i]) or closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    active = False
                    entry_bar = -1
                    fixed_stop = np.nan
                else:
                    raw[i] = 1
            elif eligible[i]:
                active = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
