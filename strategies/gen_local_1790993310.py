from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    path_bars: int = 10
    history_bars: int = 252
    neighbors: int = 9
    threshold_bars: int = 126
    edge_percentile: float = 0.75
    hold_bars: int = 18
    atr_bars: int = 14
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1790993310"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(max(
            params.atr_bars + 1,
            2 * params.path_bars + params.history_bars
            + params.hold_bars + params.threshold_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = params.path_bars
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        energy = returns.pow(2).rolling(length).sum().pow(0.5).replace(0, np.nan)
        displacement = returns.rolling(length).sum()
        # Antisymmetric level-two signature of the (time, log-price) path.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            weight = (lag + 0.5) / length - 0.5
            area = area + weight * returns.shift(lag)
        out["signature_area"] = area / energy
        out["displacement"] = displacement / energy
        out["efficiency"] = displacement.abs() / returns.abs().rolling(length).sum().replace(0, np.nan)
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_bars).mean()
        features = out[["signature_area", "displacement", "efficiency"]].to_numpy()
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = out["atr"].to_numpy(dtype=float)
        edge = np.full(len(data), np.nan)
        first = 2 * length + params.history_bars + params.hold_bars
        # Only matured, nonoverlapping historical paths can supply analog outcomes.
        for i in range(first, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            end = i - params.hold_bars - length
            candidates = np.arange(end - params.history_bars + 1, end + 1, length)
            valid = np.isfinite(features[candidates]).all(axis=1)
            valid &= np.isfinite(atr[candidates]) & (atr[candidates] > 0)
            valid &= np.isfinite(opening[candidates + 1]) & (opening[candidates + 1] > 0)
            valid &= np.isfinite(close[candidates + params.hold_bars])
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distance = np.square(features[candidates] - features[i]).sum(axis=1)
            nearest = candidates[np.argsort(distance)[:params.neighbors]]
            outcomes = (close[nearest + params.hold_bars] - opening[nearest + 1]) / atr[nearest]
            edge[i] = float(np.mean(outcomes))
        out["analog_edge"] = edge
        out["edge_threshold"] = out["analog_edge"].abs().shift(1).rolling(
            params.threshold_bars, min_periods=params.threshold_bars
        ).quantile(params.edge_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        edge = indicators["analog_edge"].to_numpy(dtype=float)
        threshold = indicators["edge_threshold"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if side != 0 and entry_bar == -1:
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    side = 0
                    continue
                entry_bar = i
                stop = opening[i] - side * params.stop_atr * atr[i - 1]
            if side != 0:
                stopped = np.isfinite(close[i]) and (
                    (side == 1 and close[i] < stop)
                    or (side == -1 and close[i] > stop)
                )
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
                    entry_bar = -1
                    stop = np.nan
                    continue
            elif (
                np.isfinite(edge[i]) and np.isfinite(threshold[i])
                and np.isfinite(atr[i]) and atr[i] > 0
                and abs(edge[i]) > threshold[i]
            ):
                side = 1 if edge[i] > 0 else -1
                entry_bar = -1
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
