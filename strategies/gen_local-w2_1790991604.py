from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityResetParams:
    recent_window: int = 4
    baseline_window: int = 20
    volatility_ratio: float = 0.65
    decline_threshold: float = 1.2
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[VolatilityResetParams]):
    strategy_id = "gen_local-w2_1790991604"

    @classmethod
    def params_type(cls) -> type[VolatilityResetParams]:
        return VolatilityResetParams

    def warmup_bars(self, params: VolatilityResetParams) -> int:
        return int(max(params.recent_window + params.baseline_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VolatilityResetParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        historical_returns = returns.shift(params.recent_window)
        recent_std = returns.rolling(params.recent_window, min_periods=params.recent_window).std(ddof=0)
        baseline_std = historical_returns.rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        baseline_mean = historical_returns.rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        recent_mean = returns.rolling(params.recent_window, min_periods=params.recent_window).mean()
        baseline_scale = baseline_std.replace(0.0, np.nan)
        volatility_ratio = recent_std / baseline_scale
        decline_score = baseline_mean * np.sqrt(params.baseline_window) / baseline_scale
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out = pd.DataFrame({
            "return": returns,
            "historical_return": historical_returns,
            "recent_std": recent_std,
            "baseline_std": baseline_std,
            "baseline_mean": baseline_mean,
            "recent_mean": recent_mean,
            "volatility_ratio": volatility_ratio,
            "decline_score": decline_score,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        out["candidate"] = (
            out["volatility_ratio"].le(params.volatility_ratio)
            & out["decline_score"].le(-params.decline_threshold)
            & out["recent_mean"].gt(0.0)
            & out["return"].gt(0.0)
            & out["atr"].gt(0.0)
        ).fillna(False)
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityResetParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = np.nan
        stop = np.nan
        peak_close = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if pending:
                # The previous close's decision fills at this bar's open.
                entry = opens[i]
                stop = entry - params.initial_stop_atr * atr[i - 1]
                peak_close = entry
                entry_bar = i
                active = True
                pending = False
                armed = False

            if active:
                # An intraday stop breach requests liquidation next bar.
                exit_now = lows[i] <= stop or i - entry_bar + 1 >= params.max_hold_bars
                if not exit_now:
                    peak_close = max(peak_close, closes[i])
                    armed = armed or highs[i] >= entry * (1.0 + params.breakeven_pct)
                    if armed and np.isfinite(atr[i]) and atr[i] > 0.0:
                        # New stops take effect after this close; never ratchet down.
                        stop = max(stop, entry, peak_close - params.trail_atr * atr[i])
                    exit_now = closes[i] <= stop
                if exit_now:
                    active = False
                else:
                    desired[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                desired[i] = 1
                pending = True

        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
