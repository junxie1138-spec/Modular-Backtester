from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    calibration_window: int = 126
    entry_quantile: float = 0.75
    profit_target: float = 0.025
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1790997833"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(3, params.graph_window) + max(2, params.calibration_window) + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        window = max(3, int(params.graph_window))
        calibration = max(2, int(params.calibration_window))
        quantile = float(np.clip(params.entry_quantile, 0.55, 0.95))
        close = data["close"].astype(float).where(lambda x: x > 0)
        log_close = np.log(close)
        prices = np.column_stack([
            log_close.shift(window - 1 - j).to_numpy()
            for j in range(window)
        ])
        valid = np.isfinite(prices).all(axis=1)
        adjacency = np.zeros((len(data), window, window), dtype=float)

        # Build natural visibility edges simultaneously for every trailing window.
        for left in range(window - 1):
            for right in range(left + 1, window):
                visible = valid.copy()
                for middle in range(left + 1, right):
                    fraction = (middle - left) / (right - left)
                    boundary = prices[:, left] + fraction * (prices[:, right] - prices[:, left])
                    visible &= prices[:, middle] < boundary
                adjacency[:, left, right] = visible
                adjacency[:, right, left] = visible

        degree = adjacency.sum(axis=2)
        connected_triples = (degree * (degree - 1.0)).sum(axis=1)
        closed_triples = ((adjacency @ adjacency) * adjacency).sum(axis=(1, 2))
        closure = np.divide(
            closed_triples,
            connected_triples,
            out=np.full(len(data), np.nan),
            where=valid & (connected_triples > 0),
        )

        result = pd.DataFrame(index=data.index)
        result["triangle_closure"] = closure
        result["closure_median"] = result["triangle_closure"].shift(1).rolling(
            calibration, min_periods=calibration
        ).quantile(0.5)
        result["trend"] = close / close.shift(window - 1) - 1.0
        result["trend_upper"] = result["trend"].shift(1).rolling(
            calibration, min_periods=calibration
        ).quantile(quantile)
        result["displacement"] = close / close.rolling(
            window, min_periods=window
        ).mean() - 1.0
        result["displacement_lower"] = result["displacement"].shift(1).rolling(
            calibration, min_periods=calibration
        ).quantile(1.0 - quantile)
        result["return_1"] = close / close.shift(1) - 1.0
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        ready = np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1)
        low_closure = indicators["triangle_closure"] < indicators["closure_median"]
        continuation = (
            low_closure
            & (indicators["trend"] > 0.0)
            & (indicators["trend"] >= indicators["trend_upper"])
            & (indicators["return_1"] > 0.0)
        )
        rebound = (
            ~low_closure
            & (indicators["displacement"] < 0.0)
            & (indicators["displacement"] <= indicators["displacement_lower"])
            & (indicators["return_1"] > 0.0)
        )
        entries = ((continuation | rebound).fillna(False).to_numpy() & ready)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw_signal = np.zeros(len(data), dtype=int)
        holding_limit = max(1, int(params.holding_bars))
        target = max(0.0001, float(params.profit_target))
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # Yesterday's decision fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = (
                    valid_entry
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + target)
                )
                time_expired = i - entry_bar + 1 >= holding_limit
                if not valid_entry or target_hit or time_expired:
                    active = False
                    entry_price = np.nan
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            elif entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
