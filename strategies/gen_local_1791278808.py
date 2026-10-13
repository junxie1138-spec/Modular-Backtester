from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 3
    scale_bars: int = 40
    shape_bars: int = 5
    library_bars: int = 252
    neighbors: int = 9
    outcome_bars: int = 18
    min_forecast: float = 0.008
    flow_bars: int = 8
    flow_threshold: float = 0.10
    atr_bars: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_holding: int = 20


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791278808"

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(
            params.roc_bars + params.scale_bars + 2 * params.shape_bars
            + params.outcome_bars + params.library_bars + 1,
            params.atr_bars + 1, params.flow_bars
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close / close.shift(params.roc_bars) - 1.0
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scale"] = out["acceleration"].rolling(
            params.scale_bars, min_periods=params.scale_bars
        ).std().replace(0.0, np.nan)
        out["normalized_acceleration"] = (
            out["acceleration"] / out["acceleration_scale"]
        )
        features = []
        for j in range(params.shape_bars):
            name = f"shape_{j}"
            out[name] = out["normalized_acceleration"].shift(j)
            features.append(name)
        current = out[features].to_numpy(dtype=float)
        n = len(data)
        distances = np.full((n, params.library_bars), np.inf)
        outcomes = np.full_like(distances, np.nan)
        # ponytail: bounded exhaustive search; chunk candidates if the library grows.
        for column in range(params.library_bars):
            lag = params.outcome_bars + params.shape_bars + column
            past = out[features].shift(lag).to_numpy(dtype=float)
            outcome = (
                close.shift(lag - params.outcome_bars) / close.shift(lag) - 1.0
            ).to_numpy(dtype=float)
            distance = np.mean((current - past) ** 2, axis=1)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[valid, column] = distance[valid]
            outcomes[valid, column] = outcome[valid]
        k = min(params.neighbors, params.library_bars)
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        selected = np.take_along_axis(outcomes, nearest, axis=1)
        ready = np.isfinite(distances).sum(axis=1) == params.library_bars
        forecast = np.mean(selected, axis=1)
        out["analog_forecast"] = np.where(ready, forecast, np.nan)
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["close_location"] = (
            (2.0 * close - data["high"] - data["low"]) / span
        ).clip(-1.0, 1.0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = out["close_location"] * volume
        out["volume_pressure"] = (
            out["signed_volume"].rolling(params.flow_bars).sum()
            / volume.rolling(params.flow_bars).sum().replace(0.0, np.nan)
        )
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        forecast = indicators["analog_forecast"]
        pressure = indicators["volume_pressure"]
        wants = np.select([
            (forecast > params.min_forecast) & (pressure > params.flow_threshold),
            (forecast < -params.min_forecast) & (pressure < -params.flow_threshold)
        ], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        entry = best_profit = stop_profit = 0.0
        age = 0
        armed = False
        for i in range(len(data)):
            if side:
                if pending:
                    # Yesterday's decision fills at today's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        continue
                    entry = opens[i]
                    stop_profit = -params.trail_atr * atrs[i - 1]
                    best_profit = 0.0
                    age = 0
                    armed = False
                    pending = False
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    side = 0
                else:
                    profit = side * (closes[i] - entry)
                    best_profit = max(best_profit, profit)
                    armed = armed or best_profit >= params.breakeven_pct * entry
                    if armed:
                        # Profit-coordinate stop only rises, also for shorts.
                        stop_profit = max(stop_profit, 0.0)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop_profit = max(
                                stop_profit, best_profit - params.trail_atr * atrs[i]
                            )
                    # Close-based exit decision; execution remains next bar.
                    if profit <= stop_profit or age >= params.max_holding:
                        side = 0
            elif (
                wants[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atrs[i]) and atrs[i] > 0
            ):
                side = int(wants[i])
                pending = True
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = AnalogParams()
    for side in (1, -1):
        index = pd.date_range("2020-01-01", periods=5)
        closes = [100.0, 100.0 + side * 3.0, 100.0, 100.0, 100.0]
        data = pd.DataFrame({
            "open": 100.0, "high": np.maximum(closes, 100.0) + 1.0,
            "low": np.minimum(closes, 100.0) - 1.0,
            "close": closes, "volume": 1000.0
        }, index=index)
        derived = pd.DataFrame({
            "analog_forecast": side * 0.02,
            "volume_pressure": side * 0.2, "atr": 1.0
        }, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params)
        assert result.data["signal"].iloc[:4].tolist() == [0, side, side, 0]
        params.max_holding = 2
        data["close"] = 100.0
        result = GeneratedStrategy.generate_signals(data, derived, None, params)
        assert result.data["signal"].iloc[:4].tolist() == [0, side, side, 0]
        params.max_holding = 20


if __name__ == "__main__":
    _self_check()
