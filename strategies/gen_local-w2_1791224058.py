from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    area_window: int = 8
    area_threshold: float = 0.03
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791224058"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.area_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        gap = np.log(data["open"] / data["close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        day = np.log(data["close"] / data["open"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gap_path, day_path = gap.cumsum(), day.cumsum()
        cross = gap_path.shift(1).fillna(0.0) * day - day_path.shift(1).fillna(0.0) * gap
        w = params.area_window
        area = 0.5 * (
            cross.rolling(w).sum()
            - gap_path.shift(w).fillna(0.0) * day.rolling(w).sum()
            + day_path.shift(w).fillna(0.0) * gap.rolling(w).sum()
        )
        length = (gap.abs() + day.abs()).rolling(w).sum()
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["area"] = (area / length.pow(2).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["gap_sum"] = gap.rolling(w).sum().fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        valid = (indicators["area"] > params.area_threshold) & (indicators["atr"] > 0)
        long_ready = valid & (indicators["gap_sum"] > 0)
        short_ready = valid & (indicators["gap_sum"] < 0)
        long_entry = (long_ready & long_ready.shift(1, fill_value=False) & ~long_ready.shift(2, fill_value=False)).to_numpy()
        short_entry = (short_ready & short_ready.shift(1, fill_value=False) & ~short_ready.shift(2, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        side, held, entry, stop = 0, 0, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            if side:
                if held == 0:
                    entry = float(opens[i])
                held += 1
                if not armed and side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                    armed, stop = True, entry
                if armed:
                    candidate = closes[i] - side * params.trail_atr * atr[i]
                    stop = max(stop, candidate) if side == 1 else min(stop, candidate)
                if (armed and side * (closes[i] - stop) <= 0) or held >= params.max_hold:
                    side, held, armed = 0, 0, False
            elif long_entry[i]:
                side = 1
            elif short_entry[i]:
                side = -1
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
