from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    gap_window: int = 5
    acceleration_lag: int = 3
    volatility_window: int = 20
    atr_window: int = 14
    regime_window: int = 200
    acceleration_threshold: float = 0.5
    erosion_fraction: float = 0.35
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local-w2_1790995098"

    @classmethod
    def params_type(cls):
        return GapAccelerationParams

    def warmup_bars(self, params: GapAccelerationParams) -> int:
        return int(max(200, params.regime_window,
                       params.gap_window + params.acceleration_lag + 2,
                       params.volatility_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["gap_log"] = np.log(data["open"] / previous_close)
        out["intraday_log"] = np.log(data["close"] / data["open"])
        out["gap_sum"] = out["gap_log"].rolling(params.gap_window).sum()
        out["intraday_sum"] = out["intraday_log"].rolling(params.gap_window).sum()
        out["gap_roc"] = np.expm1(out["gap_sum"])
        out["acceleration"] = out["gap_roc"].diff(params.acceleration_lag) / params.acceleration_lag
        out["gap_volatility"] = out["gap_log"].rolling(params.volatility_window).std(ddof=0)
        scale = (out["gap_volatility"] * np.sqrt(params.gap_window)
                 / params.acceleration_lag).replace(0.0, np.nan)
        out["acceleration_score"] = out["acceleration"] / scale
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["regime_ma"] = data["close"].rolling(max(200, params.regime_window)).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapAccelerationParams,
    ) -> SignalFrame:
        score = indicators["acceleration_score"]
        ready = np.isfinite(indicators).all(axis=1)
        ready &= (indicators["atr"] > 0.0) & np.isfinite(score.shift(1))
        long_entry = (
            ready & (data["close"] > indicators["regime_ma"])
            & (indicators["gap_roc"] > 0.0)
            & (score > params.acceleration_threshold)
            & (score.shift(1) <= params.acceleration_threshold)
            & (indicators["intraday_sum"] >= -params.erosion_fraction * indicators["gap_sum"])
        )
        short_entry = (
            ready & (data["close"] < indicators["regime_ma"])
            & (indicators["gap_roc"] < 0.0)
            & (score < -params.acceleration_threshold)
            & (score.shift(1) >= -params.acceleration_threshold)
            & (indicators["intraday_sum"] <= -params.erosion_fraction * indicators["gap_sum"])
        )
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_bar = 0
        fixed_stop = np.nan

        for i in range(len(data)):
            # A decision from the preceding close fills at this bar's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = pending
                    entry_bar = i
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                pending = 0
            if position:
                stop_hit = (
                    np.isfinite(closes[i])
                    and position * (closes[i] - fixed_stop) < 0.0
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    position = 0
                    continue
                signals[i] = position
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                pending = int(candidates[i])
                signals[i] = pending

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
