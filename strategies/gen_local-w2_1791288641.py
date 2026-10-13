from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTransportParams:
    transport_window: int = 3
    percentile_window: int = 60
    entry_percentile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 0.8
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeTransportParams]):
    strategy_id = "gen_local-w2_1791288641"

    @classmethod
    def params_type(cls) -> type[RangeTransportParams]:
        return RangeTransportParams

    def warmup_bars(self, params: RangeTransportParams) -> int:
        return int(max(params.transport_window + params.percentile_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeTransportParams) -> pd.DataFrame:
        high_move = data["high"].diff()
        low_move = data["low"].diff()
        upward = pd.concat([high_move.clip(lower=0), low_move.clip(lower=0)], axis=1).min(axis=1, skipna=False)
        downward = pd.concat([(-high_move).clip(lower=0), (-low_move).clip(lower=0)], axis=1).min(axis=1, skipna=False)
        transport = upward - downward
        edge_motion = (high_move.abs() + low_move.abs()) / 2.0
        numerator = transport.rolling(params.transport_window, min_periods=params.transport_window).sum()
        denominator = edge_motion.rolling(params.transport_window, min_periods=params.transport_window).sum()
        efficiency = numerator.div(denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        threshold = efficiency.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            "high_move": high_move,
            "low_move": low_move,
            "upward": upward,
            "downward": downward,
            "transport": transport,
            "edge_motion": edge_motion,
            "transport_sum": numerator,
            "motion_sum": denominator,
            "efficiency": efficiency,
            "threshold": threshold,
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeTransportParams) -> SignalFrame:
        ready = indicators[["efficiency", "threshold", "atr"]].notna().all(axis=1)
        entries = (
            ready
            & (indicators["atr"] > 0)
            & (indicators["efficiency"] > indicators["threshold"])
            & (indicators["efficiency"] > 0)
            & (indicators["high_move"] > 0)
            & (indicators["low_move"] > 0)
            & (data["close"] > data["open"])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        trailing_stop = np.nan
        hold_limit = max(1, min(2, int(params.max_hold_bars)))

        for i in range(len(data)):
            if active:
                held += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]):
                    active = False
                else:
                    high_water = closes[i] if not np.isfinite(high_water) else max(high_water, closes[i])
                    candidate_stop = high_water - params.stop_atr * atrs[i]
                    trailing_stop = candidate_stop if not np.isfinite(trailing_stop) else max(trailing_stop, candidate_stop)
                    if closes[i] <= trailing_stop or held >= hold_limit:
                        active = False
                signals[i] = int(active)
            elif entries[i] and np.isfinite(closes[i]):
                active = True
                held = 0
                high_water = np.nan
                trailing_stop = np.nan
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
