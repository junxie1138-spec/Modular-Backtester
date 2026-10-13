from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    graph_window: int = 12
    history_bars: int = 252
    neighbors: int = 7
    hold_bars: int = 7
    min_forecast: float = 0.003
    min_agreement: float = 0.70
    pressure_bars: int = 3
    pressure_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1790885322"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(max(params.history_bars + params.graph_window + params.hold_bars,
                       params.pressure_bars))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        w, h, history, k = (params.graph_window, params.hold_bars,
                            params.history_bars, params.neighbors)
        if (w < 3 or h < 1 or k < 1 or history < w + h + k * h
                or params.pressure_bars < 1 or params.min_forecast < 0
                or not 0.5 < params.min_agreement <= 1
                or not 0 <= params.pressure_threshold <= 1):
            raise ValueError("Invalid shape, history, holding, or threshold parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        n = len(data)
        prices = out["log_close"].to_numpy(dtype=float)
        pairs = [(a, b) for a in range(w) for b in range(a + 2, w)]
        signatures = np.full((n, len(pairs)), np.nan)
        if n >= w:
            windows = np.lib.stride_tricks.sliding_window_view(prices, w)
            valid = np.isfinite(windows).all(axis=1)
            for j, (a, b) in enumerate(pairs):
                # Horizontal visibility: every intervening close is below both endpoints.
                visible = windows[:, a + 1:b].max(axis=1) < np.minimum(
                    windows[:, a], windows[:, b])
                signatures[w - 1:, j] = np.where(valid, visible.astype(float), np.nan)
        out = pd.concat([out, pd.DataFrame(
            signatures, index=data.index,
            columns=[f"edge_{a}_{b}" for a, b in pairs])], axis=1)
        forecasts = np.full(n, np.nan)
        agreements = np.full(n, np.nan)
        # ponytail: exact bounded-history graph scans; index signatures if history grows substantially.
        for t in range(GeneratedStrategy.warmup_bars(params) - 1, n):
            if not np.isfinite(signatures[t]).all():
                continue
            # Outcome endpoints precede the current graph window; sample at holding-period spacing.
            candidates = np.arange(max(w - 1, t - history), t - w - h + 1, h)
            usable = (np.isfinite(signatures[candidates]).all(axis=1)
                      & np.isfinite(prices[candidates])
                      & np.isfinite(prices[candidates + h]))
            candidates = candidates[usable]
            if len(candidates) < k:
                continue
            distances = np.mean(signatures[candidates] != signatures[t], axis=1)
            selected = candidates[np.argsort(distances, kind="stable")[:k]]
            outcomes = prices[selected + h] - prices[selected]
            forecast = float(np.median(outcomes))
            forecasts[t] = forecast
            agreements[t] = np.mean(outcomes * np.sign(forecast) > 0)
        out["analog_forecast"] = forecasts
        out["analog_agreement"] = agreements
        spread = (data["high"] - data["low"]).where(data["high"] > data["low"])
        location = ((2 * data["close"] - data["high"] - data["low"]) / spread).clip(-1, 1)
        location = location.where(spread.notna(), 0.0).where(data["close"].notna())
        volume = data["volume"].where(data["volume"] >= 0)
        out["close_location"] = location
        out["signed_volume"] = location * volume
        out["volume_sum"] = volume.rolling(params.pressure_bars).sum()
        out["pressure"] = (out["signed_volume"].rolling(params.pressure_bars).sum()
                           / out["volume_sum"].where(out["volume_sum"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        forecast = indicators["analog_forecast"]
        pressure = indicators["pressure"]
        qualified = indicators["analog_agreement"] >= params.min_agreement
        long_entry = qualified & (forecast > params.min_forecast) & (pressure > params.pressure_threshold)
        short_entry = qualified & (forecast < -params.min_forecast) & (pressure < -params.pressure_threshold)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        targets = np.zeros(len(data), dtype=int)
        position, entered = 0, -1
        for i, entry in enumerate(entries):
            if position and i - entered == params.hold_bars:
                position = 0
            elif position == 0 and entry:
                position, entered = int(entry), i
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ShapeParams(hold_bars=7)
    index = pd.date_range("2000-01-01", periods=18)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "analog_forecast": [0.02] + [-0.02] * 17,
        "analog_agreement": 1.0,
        "pressure": [0.5] + [-0.5] * 17,
    }, index=index)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].tolist() == [0] + [1] * 7 + [0] + [-1] * 7 + [0, -1]
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
