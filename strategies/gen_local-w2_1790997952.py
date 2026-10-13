from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    area_threshold: float = 0.20
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local-w2_1790997952"

    @classmethod
    def params_type(cls) -> type[SignatureRecoveryParams]:
        return SignatureRecoveryParams

    @staticmethod
    def warmup_bars(params: SignatureRecoveryParams) -> int:
        # Twenty-bar participation baseline, one difference, four increments.
        return 24

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            high - low,
            (high - out["previous_close"]).abs(),
            (low - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean().replace(0.0, np.nan)
        out["volume_mean"] = volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
        out["volume_ratio"] = volume / out["volume_mean"]
        spread = (high - low).replace(0.0, np.nan)
        out["closing_location"] = ((2.0 * close - high - low) / spread).where(spread.notna(), 0.0)
        out["pressure"] = out["closing_location"] * out["volume_ratio"]
        out["price_increment"] = close.diff() / out["atr"].shift(1)
        out["pressure_increment"] = out["pressure"].diff()
        # Antisymmetric second-level signature of four chronological increments.
        area = pd.Series(0.0, index=data.index)
        for older in range(3, 0, -1):
            for newer in range(older - 1, -1, -1):
                area = area + 0.5 * (
                    out["price_increment"].shift(older) * out["pressure_increment"].shift(newer)
                    - out["pressure_increment"].shift(older) * out["price_increment"].shift(newer)
                )
        out["signature_area"] = area
        out["prior_high"] = close.rolling(20, min_periods=20).max().shift(1)
        out["drawdown_atr"] = (out["prior_high"] - close) / out["atr"]
        out["prior_rebound"] = close.shift(1) - close.shift(3)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureRecoveryParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators.notna().all(axis=1)
        recovering = (
            ready
            & (indicators["drawdown_atr"] >= 1.0)
            & (indicators["price_increment"] > 0.0)
            & (indicators["pressure"] > 0.0)
            & (indicators["pressure_increment"] > 0.0)
            & (indicators["signature_area"] < -params.area_threshold)
        )
        failing = (
            ready
            & (indicators["drawdown_atr"] >= 1.0)
            & (indicators["prior_rebound"] > 0.0)
            & (indicators["price_increment"] < 0.0)
            & (indicators["pressure"] < 0.0)
            & (indicators["pressure_increment"] < 0.0)
            & (indicators["signature_area"] > params.area_threshold)
        )
        entries = np.select([recovering, failing], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            if position != 0:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    position = 0
                else:
                    # First held bar is the bar after the entry decision.
                    high_water = close[i] if not np.isfinite(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # Highest-close trailing exit applies to both directions;
                    # for shorts, a falling-close trigger realizes the decline.
                    if close[i] <= stop or age >= 2:
                        position = 0
                signals[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                stop = -np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
