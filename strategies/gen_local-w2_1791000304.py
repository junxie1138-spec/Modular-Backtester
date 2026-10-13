from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    signature_bars: int = 5
    volume_bars: int = 20
    drawdown_bars: int = 20
    pressure_bars: int = 3
    area_threshold: float = 0.08
    pressure_threshold: float = 0.20
    min_drawdown: float = 0.015
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791000304"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.volume_bars + params.signature_bars,
                       params.signature_bars + 1,
                       params.drawdown_bars,
                       params.pressure_bars) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).clip(lower=0)
        baseline = volume.rolling(params.volume_bars,
                                  min_periods=params.volume_bars).mean()
        out["price_step"] = np.log(close).diff()
        out["volume_step"] = volume.div(baseline.where(baseline > 0))

        # Second-order antisymmetric signature, with the window origin reset.
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        price_length = pd.Series(0.0, index=data.index)
        for lag in range(params.signature_bars - 1, -1, -1):
            dx = out["price_step"].shift(lag)
            dy = out["volume_step"].shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
            price_length = price_length + dx.abs()
        out["signature_price_displacement"] = x
        out["signature_volume_displacement"] = y
        out["signature_price_length"] = price_length
        out["signature_area"] = area
        scale = price_length * y
        out["normalized_area"] = area.div(scale.where(scale > 0))

        peak = close.rolling(params.drawdown_bars,
                             min_periods=params.drawdown_bars).max()
        out["rolling_peak"] = peak
        out["drawdown"] = close.div(peak) - 1.0
        spread = (data["high"] - data["low"]).astype(float)
        location = (2.0 * close - data["high"] - data["low"]).div(
            spread.where(spread > 0))
        location = location.mask(spread.eq(0), 0.0).clip(-1.0, 1.0)
        out["close_location"] = location
        out["pressure_numerator"] = (location * volume).rolling(
            params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure_volume"] = volume.rolling(
            params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure"] = out["pressure_numerator"].div(
            out["pressure_volume"].where(out["pressure_volume"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: RecoveryParams) -> SignalFrame:
        depressed = indicators["drawdown"].le(-params.min_drawdown)
        long_entry = (depressed
                      & indicators["normalized_area"].le(-params.area_threshold)
                      & indicators["pressure"].ge(params.pressure_threshold))
        short_entry = (depressed
                       & indicators["normalized_area"].ge(params.area_threshold)
                       & indicators["pressure"].le(-params.pressure_threshold))
        candidates = np.select([long_entry.fillna(False),
                                short_entry.fillna(False)], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        earliest = GeneratedStrategy.warmup_bars(params) - 1

        for i in range(len(data)):
            if position != 0:
                # Yesterday's entry decision fills at today's open.
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        entry_price = np.nan
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or expired:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
                    # Zero decided at this close executes at the next open.
                    signals[i] = 0
                    continue
            elif i >= earliest and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_bar = -1
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
