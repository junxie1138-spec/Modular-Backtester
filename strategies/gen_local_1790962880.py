from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeEchoParams:
    correlation_window: int = 63
    volume_window: int = 20
    atr_window: int = 14
    min_events: int = 8
    volume_threshold: float = 1.15
    correlation_threshold: float = 0.15
    stop_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolumeEchoParams]):
    strategy_id = "gen_local_1790962880"

    @classmethod
    def params_type(cls) -> type[VolumeEchoParams]:
        return VolumeEchoParams

    def warmup_bars(self, params: VolumeEchoParams) -> int:
        return max(params.correlation_window + 2, params.volume_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: VolumeEchoParams) -> pd.DataFrame:
        if not (params.correlation_window >= 3 and params.volume_window >= 2
                and params.atr_window >= 1 and 2 <= params.min_events <= params.correlation_window
                and params.volume_threshold > 0 and 0 <= params.correlation_threshold < 1
                and params.stop_atr > 0 and params.hold_bars >= 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        previous = close.shift(1)
        returns = (close / previous.where(previous > 0) - 1).replace([np.inf, -np.inf], np.nan)
        baseline = data["volume"].rolling(params.volume_window).mean().shift(1)
        relative_volume = (data["volume"] / baseline.where(baseline > 0)).replace([np.inf, -np.inf], np.nan)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        x = returns.shift(1)
        y = returns
        event = (relative_volume.shift(1) >= params.volume_threshold) & x.notna() & y.notna()
        window = params.correlation_window
        count = event.astype(float).rolling(window).sum()
        denominator = count.where(count >= params.min_events)
        ex = x.where(event, 0.0).rolling(window).sum() / denominator
        ey = y.where(event, 0.0).rolling(window).sum() / denominator
        exx = (x * x).where(event, 0.0).rolling(window).sum() / denominator
        eyy = (y * y).where(event, 0.0).rolling(window).sum() / denominator
        exy = (x * y).where(event, 0.0).rolling(window).sum() / denominator
        variance_x = (exx - ex * ex).clip(lower=0)
        variance_y = (eyy - ey * ey).clip(lower=0)
        covariance = exy - ex * ey
        scale = np.sqrt(variance_x * variance_y)
        correlation = (covariance / scale.where(scale > 0)).clip(-1, 1)
        slope = (covariance / variance_x.where(variance_x > 0)).clip(0, 2)
        forecast = slope * returns
        atr_fraction = atr / close.where(close > 0)
        conviction = (correlation.clip(lower=0) * forecast / atr_fraction.where(atr_fraction > 0) / 0.5)
        conviction = conviction.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
        entry = ((relative_volume >= params.volume_threshold)
                 & (returns > 0) & (correlation >= params.correlation_threshold)
                 & (conviction > 0) & (atr > 0) & (close > 0))
        return pd.DataFrame({
            "return": returns, "relative_volume": relative_volume, "true_range": true_range,
            "atr": atr, "lagged_return": x, "event": event, "event_count": count,
            "mean_x": ex, "mean_y": ey, "mean_xx": exx, "mean_yy": eyy,
            "mean_xy": exy, "variance_x": variance_x, "variance_y": variance_y,
            "covariance": covariance, "correlation": correlation, "slope": slope,
            "forecast": forecast, "atr_fraction": atr_fraction,
            "conviction": conviction, "entry": entry.fillna(False),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VolumeEchoParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        allocation = 0.2
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        convictions = indicators["conviction"].fillna(0.0).to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # The high-water mark starts on the fill bar, after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.stop_atr * atr)
                if not valid or close <= stop or age >= params.hold_bars:
                    active = False
                    continue
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                allocation = 0.2 + 0.75 * float(np.clip(convictions[i], 0, 1))
            if active:
                signals[i] = 1
                sizes[i] = allocation
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 105., 104., 101., 100., 99.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 10., 10., 1., 1.],
        "entry": [True, False, False, False, False, False],
        "conviction": [0.5] * 6,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, VolumeEchoParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.575)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
