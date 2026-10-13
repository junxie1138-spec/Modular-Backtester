from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConcentrationParams:
    regime_window: int = 12
    volatility_window: int = 20
    momentum_window: int = 3
    atr_window: int = 14
    distributed_threshold: float = 0.65
    concentrated_threshold: float = 0.45
    fresh_shock_share: float = 0.22
    entry_threshold: float = 0.8
    trail_atr: float = 1.5
    hold_bars: int = 4
    minimum_size: float = 0.25
    full_size_score: float = 2.5


class GeneratedStrategy(BaseStrategy[ConcentrationParams]):
    strategy_id = "gen_local_1790999678"

    @classmethod
    def params_type(cls) -> type[ConcentrationParams]:
        return ConcentrationParams

    def warmup_bars(self, params: ConcentrationParams) -> int:
        return int(max(params.regime_window, params.volatility_window,
                       params.momentum_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: ConcentrationParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        absolute_total = returns.abs().rolling(params.regime_window, min_periods=params.regime_window).sum()
        squared_total = returns.pow(2).rolling(params.regime_window, min_periods=params.regime_window).sum()
        participation = absolute_total.pow(2).div(params.regime_window * squared_total.replace(0.0, np.nan)).clip(0.0, 1.0)
        shock_share = returns.abs().div(absolute_total.replace(0.0, np.nan))
        volatility = returns.rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0).replace(0.0, np.nan)
        momentum = returns.rolling(params.momentum_window, min_periods=params.momentum_window).sum().div(volatility * np.sqrt(params.momentum_window))
        impulse = returns.div(volatility)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        result = pd.DataFrame({
            "return": returns,
            "participation": participation,
            "shock_share": shock_share,
            "volatility": volatility,
            "momentum": momentum,
            "impulse": impulse,
            "true_range": true_range,
            "atr": atr
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        result["ready"] = result.notna().all(axis=1) & result["atr"].gt(0.0)
        result = result.fillna(0.0)
        distributed = result["participation"].ge(params.distributed_threshold)
        concentrated = result["participation"].le(params.concentrated_threshold) & result["shock_share"].ge(params.fresh_shock_share)
        result["score"] = pd.Series(np.where(distributed, result["momentum"], np.where(concentrated, -result["impulse"], 0.0)), index=data.index).where(result["ready"], 0.0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ConcentrationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        state = 0
        age = 0
        peak = -np.inf
        trough = np.inf
        trailing_stop = -np.inf
        short_cover = np.inf
        entry_size = 1.0
        holding_limit = max(1, int(params.hold_bars))
        minimum_size = float(np.clip(params.minimum_size, 0.01, 1.0))
        size_scale = max(float(params.full_size_score), 1e-12)
        stop_multiple = max(float(params.trail_atr), 1e-12)

        for i in range(count):
            if state != 0:
                age += 1
                if not ready[i] or not np.isfinite(closes[i]):
                    state = 0
                else:
                    # Every position tracks its highest observed close since entry.
                    peak = max(peak, closes[i])
                    trailing_stop = max(trailing_stop, peak - stop_multiple * atrs[i])
                    # Shorts also cover adverse rebounds from their lowest close.
                    trough = min(trough, closes[i])
                    short_cover = min(short_cover, trough + stop_multiple * atrs[i])
                    high_water_exit = closes[i] <= trailing_stop
                    adverse_short_exit = state == -1 and closes[i] >= short_cover
                    if high_water_exit or adverse_short_exit or age >= holding_limit:
                        state = 0
                if state != 0:
                    signals[i] = state
                    sizes[i] = entry_size
                # An exit decision remains flat for this bar; no immediate reentry.
                continue

            if ready[i] and np.isfinite(closes[i]) and abs(scores[i]) >= params.entry_threshold and scores[i] != 0.0:
                state = 1 if scores[i] > 0.0 else -1
                entry_size = float(np.clip(abs(scores[i]) / size_scale, minimum_size, 1.0))
                age = 0
                peak = -np.inf
                trough = np.inf
                trailing_stop = -np.inf
                short_cover = np.inf
                signals[i] = state
                sizes[i] = entry_size
                # Tracking starts next bar, when this entry instruction fills.

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
