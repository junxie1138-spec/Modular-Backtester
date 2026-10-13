from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    trend_span: int = 36
    slope_span: int = 8
    vol_span: int = 10
    percentile_window: int = 126
    innovation_percentile: float = 0.80
    volatility_percentile: float = 0.65
    atr_span: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791296761"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.percentile_window + params.vol_span + 1, params.atr_span)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        log_close = np.log(data["close"].clip(lower=1e-12))
        latent = log_close.ewm(span=params.trend_span, adjust=False).mean()
        slope = latent.diff()
        forecast = slope.ewm(span=params.slope_span, adjust=False).mean().shift(1)
        volatility = log_close.diff().rolling(params.vol_span, min_periods=params.vol_span).std()
        innovation = (slope - forecast).div(volatility.replace(0, np.nan))
        innovation_cut = innovation.abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.innovation_percentile)
        volatility_cut = volatility.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.volatility_percentile)
        previous_close = data["close"].shift(1).fillna(data["close"])
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_span, min_periods=params.atr_span).mean()
        return pd.DataFrame({"innovation": innovation.fillna(0), "innovation_cut": innovation_cut.fillna(np.inf), "volatility": volatility.fillna(0), "volatility_cut": volatility_cut.fillna(np.inf), "atr": atr.fillna(0)}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        side = pending = 0
        entry = stop = peak = trough = 0.0
        bars = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            high = float(data["high"].iloc[i])
            low = float(data["low"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if pending:
                side, pending = pending, 0
                entry = float(data["open"].iloc[i])
                peak = trough = entry
                stop = entry - side * params.trail_atr * atr
                bars = 0
            if side:
                bars += 1
                if side == 1:
                    peak = max(peak, high)
                    if peak >= entry * (1 + params.breakeven_pct):
                        stop = max(stop, entry, peak - params.trail_atr * atr)
                    exit_now = close <= stop
                else:
                    trough = min(trough, low)
                    if trough <= entry * (1 - params.breakeven_pct):
                        stop = min(stop, entry, trough + params.trail_atr * atr)
                    exit_now = close >= stop
                if exit_now or bars >= params.max_hold:
                    side = 0
                signals[i] = side
                continue
            innovation = float(indicators["innovation"].iloc[i])
            cut = float(indicators["innovation_cut"].iloc[i])
            if atr > 0 and abs(innovation) > cut:
                regime = 1 if indicators["volatility"].iloc[i] <= indicators["volatility_cut"].iloc[i] else -1
                pending = regime * (1 if innovation > 0 else -1)
                signals[i] = pending
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
