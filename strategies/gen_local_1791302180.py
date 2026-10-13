from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BarrierAnalogParams:
    ma_window: int = 20
    analog_window: int = 252
    barrier_pct: float = 0.01
    edge_threshold: float = 0.18


class GeneratedStrategy(BaseStrategy[BarrierAnalogParams]):
    strategy_id = "gen_local_1791302180"

    @classmethod
    def params_type(cls) -> type[BarrierAnalogParams]:
        return BarrierAnalogParams

    def warmup_bars(self, params: BarrierAnalogParams) -> int:
        return int(params.ma_window + params.analog_window + 8)

    def indicators(self, data: pd.DataFrame, params: BarrierAnalogParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        std = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        z = ((close - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        prices = close.to_numpy()
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        n = len(data)
        horizon = 4
        shape = 4
        outcome = np.zeros(n, dtype=float)

        for j in range(n - horizon):
            upper = prices[j] * (1.0 + params.barrier_pct)
            lower = prices[j] * (1.0 - params.barrier_pct)
            for day in range(j + 1, j + horizon + 1):
                up = highs[day] >= upper
                down = lows[day] <= lower
                if up or down:
                    outcome[j] = 0.0 if up and down else (1.0 if up else -1.0)
                    break

        edge = np.zeros(n, dtype=float)
        first = params.ma_window + shape - 2
        offsets = np.arange(shape - 1, -1, -1)
        for i in range(self.warmup_bars(params), n):
            candidates = np.arange(max(first, i - horizon - params.analog_window + 1), i - horizon + 1)
            if len(candidates) < 40:
                continue
            history = z[candidates[:, None] - offsets]
            distance = np.mean((history - z[i - offsets]) ** 2, axis=1)
            nearest = np.argpartition(distance, min(11, len(candidates)) - 1)[:11]
            edge[i] = outcome[candidates[nearest]].mean()

        return pd.DataFrame({"edge": edge}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BarrierAnalogParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, edge in enumerate(indicators["edge"].to_numpy()):
            if position == 0:
                if edge > params.edge_threshold:
                    position = 1
                elif edge < -params.edge_threshold:
                    position = -1
            elif position == 1 and edge < -params.edge_threshold:
                position = 0
            elif position == -1 and edge > params.edge_threshold:
                position = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
