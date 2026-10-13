from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HistoricalShapeParams:
    drawdown_lookback: int = 63
    shape_bars: int = 12
    forecast_bars: int = 18
    analogs: int = 7
    min_depth: float = 0.04
    edge_threshold: float = 0.15
    atr_lookback: int = 14
    breakeven_trigger: float = 0.025
    trail_atr: float = 2.5
    max_hold_bars: int = 22


class GeneratedStrategy(BaseStrategy[HistoricalShapeParams]):
    strategy_id = "gen_local_1791223023"

    @classmethod
    def params_type(cls) -> type[HistoricalShapeParams]:
        return HistoricalShapeParams

    def warmup_bars(self, params: HistoricalShapeParams) -> int:
        return int(max(
            params.drawdown_lookback + params.shape_bars - 1,
            params.atr_lookback,
            params.shape_bars + params.forecast_bars + params.analogs,
        ))

    def indicators(
        self, data: pd.DataFrame, params: HistoricalShapeParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        peak = close.rolling(
            params.drawdown_lookback, min_periods=1
        ).max()
        drawdown = (close / peak - 1.0).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)

        previous_close = close.shift(1).fillna(close)
        true_range = pd.concat(
            [
                data["high"].astype(float) - data["low"].astype(float),
                (data["high"].astype(float) - previous_close).abs(),
                (data["low"].astype(float) - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(
            params.atr_lookback, min_periods=1
        ).mean().fillna(0.0)

        depth = drawdown.to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        count = len(data)
        width = max(2, params.shape_bars)
        shapes = np.zeros((count, width), dtype=float)
        valid_shape = np.zeros(count, dtype=bool)

        for end in range(width - 1, count):
            path = depth[end - width + 1 : end + 1]
            centered = path - path.mean()
            scale = max(float(centered.std()), 1e-12)
            shapes[end] = centered / scale
            valid_shape[end] = True

        scores = np.zeros(count, dtype=float)
        first_endpoint = width - 1
        first_signal_bar = first_endpoint + params.forecast_bars

        for current in range(first_signal_bar, count):
            last_endpoint = current - params.forecast_bars
            endpoints = np.arange(first_endpoint, last_endpoint + 1)
            endpoints = endpoints[valid_shape[endpoints]]
            if endpoints.size == 0:
                continue

            shape_distance = np.mean(
                (shapes[endpoints] - shapes[current]) ** 2, axis=1
            )
            depth_scale = max(abs(depth[current]), params.min_depth, 1e-6)
            depth_distance = (
                (depth[endpoints] - depth[current]) / depth_scale
            ) ** 2
            distance = shape_distance + 0.5 * depth_distance
            neighbor_count = min(max(1, params.analogs), endpoints.size)
            nearest = np.argpartition(distance, neighbor_count - 1)[
                :neighbor_count
            ]

            votes = np.zeros(neighbor_count, dtype=float)
            weights = 1.0 / (distance[nearest] + 0.05)
            for position, endpoint in enumerate(endpoints[nearest]):
                future = (
                    prices[endpoint + 1 : endpoint + params.forecast_bars + 1]
                    / prices[endpoint]
                    - 1.0
                )
                upside = max(float(future.max()), 0.0)
                downside = max(float(-future.min()), 0.0)
                votes[position] = (upside - downside) / (
                    upside + downside + 1e-12
                )

            scores[current] = float(
                np.average(votes, weights=weights)
            )

        return pd.DataFrame(
            {
                "drawdown": drawdown,
                "atr": atr,
                "analog_edge": pd.Series(
                    scores, index=data.index
                ).replace([np.inf, -np.inf], 0.0).fillna(0.0),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HistoricalShapeParams,
    ) -> SignalFrame:
        edge = indicators["analog_edge"].fillna(0.0)
        stressed = indicators["drawdown"].fillna(0.0) <= -params.min_depth
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[stressed & (edge >= params.edge_threshold)] = 1
        raw.loc[stressed & (edge <= -params.edge_threshold)] = -1
        confirmed = raw.where((raw == raw.shift(1)) & raw.ne(0), 0).astype(int)

        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)

        position = 0
        entry_price = 0.0
        extreme = 0.0
        stop = 0.0
        active = False
        held = 0

        for i in range(len(data)):
            if position == 0:
                if entries[i] != 0:
                    position = int(entries[i])
                    entry_price = close[i]
                    extreme = close[i]
                    stop = close[i]
                    active = False
                    held = 0
                positions[i] = position
                continue

            held += 1
            exit_now = held >= params.max_hold_bars or entries[i] == -position

            if position == 1:
                extreme = max(extreme, high[i])
                if not active and high[i] >= entry_price * (1.0 + params.breakeven_trigger):
                    active = True
                    stop = entry_price
                if active:
                    stop = max(stop, extreme - params.trail_atr * atr[i])
                    exit_now = exit_now or low[i] <= stop
            else:
                extreme = min(extreme, low[i])
                if not active and low[i] <= entry_price * (1.0 - params.breakeven_trigger):
                    active = True
                    stop = entry_price
                if active:
                    stop = min(stop, extreme + params.trail_atr * atr[i])
                    exit_now = exit_now or high[i] >= stop

            if exit_now:
                position = 0
                active = False
                held = 0

            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(positions, index=data.index, dtype=int)
        df["size"] = (
            0.5 + 0.5 * edge.abs().clip(lower=0.0, upper=1.0)
        ).fillna(0.5).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
