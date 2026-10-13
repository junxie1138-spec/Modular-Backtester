from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BrakingParams:
    roc_bars: int = 3
    percentile_bars: int = 80
    vol_bars: int = 20
    atr_bars: int = 14
    acceleration_quantile: float = 0.85
    target_daily_vol: float = 0.008
    breakeven_pct: float = 0.004
    atr_multiple: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[BrakingParams]):
    strategy_id = "gen_local_1791309638"

    @classmethod
    def params_type(cls):
        return BrakingParams

    @staticmethod
    def warmup_bars(params: BrakingParams) -> int:
        return int(max(params.roc_bars + 1, params.vol_bars, params.atr_bars) + params.percentile_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: BrakingParams) -> pd.DataFrame:
        if min(params.roc_bars, params.percentile_bars, params.vol_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.acceleration_quantile < 1.0:
            raise ValueError("acceleration_quantile must be between zero and one")
        if min(params.target_daily_vol, params.breakeven_pct, params.atr_multiple) <= 0:
            raise ValueError("Risk parameters must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["daily_return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["volatility"] = out["daily_return"].rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0)
        out["normalized_acceleration"] = out["acceleration"] / out["volatility"].replace(0.0, np.nan)
        out["acceleration_magnitude"] = out["normalized_acceleration"].abs()
        out["threshold"] = out["acceleration_magnitude"].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.acceleration_quantile)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["target_size"] = (params.target_daily_vol / out["volatility"].replace(0.0, np.nan)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BrakingParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0.0)
        braking = (indicators["roc"] * indicators["acceleration"] < 0.0)
        continuing_drift = (indicators["daily_return"] * indicators["roc"] > 0.0)
        extreme = indicators["acceleration_magnitude"] > indicators["threshold"]
        candidate = np.where(ready & braking & continuing_drift & extreme, np.sign(indicators["acceleration"].fillna(0.0)), 0).astype(int)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target_size = indicators["target_size"].fillna(0.01).to_numpy(dtype=float)
        position = 0
        pending = 0
        entry = 0.0
        stop = 0.0
        armed = False
        held = 0
        trade_size = 0.01
        for i in range(len(data)):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = pending
                    entry = opens[i]
                    # Signed prices let both long and short stops ratchet upward.
                    stop = position * entry - params.atr_multiple * atr[i - 1]
                    armed = False
                    held = 0
                pending = 0
            if position:
                held += 1
                adverse = lows[i] if position == 1 else -highs[i]
                opening = position * opens[i]
                stop_hit = (np.isfinite(adverse) and adverse <= stop) or (np.isfinite(opening) and opening <= stop)
                if stop_hit or held >= params.max_hold_bars or not np.isfinite(closes[i]):
                    position = 0
                    sizes[i] = trade_size
                    continue
                signed_close = position * closes[i]
                if signed_close - position * entry >= params.breakeven_pct * entry:
                    armed = True
                if armed:
                    stop = max(stop, position * entry)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, signed_close - params.atr_multiple * atr[i])
                decisions[i] = position
                sizes[i] = trade_size
            elif candidate[i]:
                pending = int(candidate[i])
                trade_size = float(target_size[i])
                decisions[i] = pending
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
