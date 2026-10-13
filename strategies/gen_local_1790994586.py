from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    neighbors: int = 15
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790994586"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 540

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        scale = out["atr"].where(out["atr"] > 0.0)
        out["slope"] = sum(
            (2.5 - lag) * close.shift(lag) for lag in range(6)
        ) / 17.5
        out["pre_slope"] = out["slope"].shift(1)
        out["innovation"] = out["slope"].diff() / scale
        out["shock_sign"] = np.sign(out["innovation"])
        out["mean_close"] = close.rolling(12, min_periods=12).mean()
        out["shape_slope"] = out["shock_sign"] * out["pre_slope"] / scale
        out["shape_bend"] = out["shock_sign"] * (
            out["slope"].shift(1) - out["slope"].shift(4)
        ) / scale
        out["shape_stretch"] = out["shock_sign"] * (
            previous - out["mean_close"].shift(1)
        ) / scale
        out["shock_amplitude"] = out["innovation"].abs()
        # A response becomes available only eight sessions after its shock.
        out["matured_excess"] = out["shock_sign"].shift(8) * (
            close - close.shift(8) - 8.0 * out["pre_slope"].shift(8)
        ) / scale.shift(8)
        features = out[
            ["shape_slope", "shape_bend", "shape_stretch", "shock_amplitude"]
        ].to_numpy(dtype=float)
        responses = out["matured_excess"].to_numpy(dtype=float)
        innovation = out["innovation"].to_numpy(dtype=float)
        baseline = (8.0 * out["pre_slope"] / scale).to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        agreement = np.full(len(data), np.nan)
        radius = np.full(len(data), np.nan)
        k = max(3, int(params.neighbors))
        # Mirror shapes by shock sign; estimate excess response over old drift.
        for i in range(self.warmup_bars(params), len(data)):
            if not np.isfinite(features[i]).all() or abs(innovation[i]) < 0.08:
                continue
            candidates = np.arange(max(20, i - 504), i - 16)
            valid = (
                np.isfinite(features[candidates]).all(axis=1)
                & np.isfinite(responses[candidates + 8])
                & (np.abs(innovation[candidates]) >= 0.08)
            )
            candidates = candidates[valid]
            if len(candidates) < k:
                continue
            distance = np.sum((features[candidates] - features[i]) ** 2, axis=1)
            chosen = np.argpartition(distance, k - 1)[:k]
            radius[i] = float(np.sqrt(distance[chosen].max()))
            predictions = baseline[i] + np.sign(innovation[i]) * responses[
                candidates[chosen] + 8
            ]
            forecast[i] = float(np.median(predictions))
            agreement[i] = float(np.mean(
                np.sign(predictions) == np.sign(forecast[i])
            ))
        out["forecast"] = forecast
        out["agreement"] = agreement
        out["analog_radius"] = radius
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        qualified = (
            indicators["forecast"].abs().ge(0.75)
            & indicators["agreement"].ge(0.7)
            & indicators["analog_radius"].le(1.5)
        )
        entries = np.where(
            qualified.fillna(False),
            np.sign(indicators["forecast"].fillna(0.0)), 0
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = low_water = floor = ceiling = 0.0
        multiple = max(0.1, float(params.trail_atr))
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                continue
            if position:
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                # The highest-close floor only ratchets upward, even as ATR changes.
                floor = max(floor, high_water - multiple * atr[i])
                ceiling = min(ceiling, low_water + multiple * atr[i])
                stop = close[i] <= floor
                if position < 0:
                    stop = stop or close[i] >= ceiling
                if stop or i - entry_bar >= 8:
                    position = 0
                    continue
            elif entries[i] and i >= self.warmup_bars(params):
                position = int(entries[i])
                entry_bar = i
                high_water = low_water = close[i]
                floor = close[i] - multiple * atr[i]
                ceiling = close[i] + multiple * atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
