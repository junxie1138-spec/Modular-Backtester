from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fast_bars: int = 5
    slow_bars: int = 30
    gap_threshold: float = 0.0007
    body_threshold: float = 0.12
    atr_bars: int = 14
    atr_multiple: float = 2.2
    breakeven_pct: float = 0.025
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791224490"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.slow_bars, params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        prior_close = data["close"].shift(1)
        gap = data["open"].div(prior_close.replace(0, np.nan)).sub(1)
        candle_range = (data["high"] - data["low"]).replace(0, np.nan)
        body = (2 * data["close"] - data["high"] - data["low"]).div(candle_range)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prior_close).abs(),
             (data["low"] - prior_close).abs()],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["gap_change"] = gap.rolling(params.fast_bars).mean() - gap.rolling(params.slow_bars).mean()
        out["body_change"] = body.rolling(params.fast_bars).mean() - body.rolling(params.slow_bars).mean()
        out["atr"] = true_range.rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        gap = indicators["gap_change"]
        body = indicators["body_change"]
        up = ((gap > params.gap_threshold)
              & (gap.shift(1) <= params.gap_threshold)
              & (body > params.body_threshold)).fillna(False).to_numpy()
        down = ((gap < -params.gap_threshold)
                & (gap.shift(1) >= -params.gap_threshold)
                & (body < -params.body_threshold)).fillna(False).to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        entry = None
        entry_bar = -1
        stop = 0.0
        active = False

        for i in range(self.warmup_bars(params), len(data)):
            previous = signal[i - 1]
            if previous == 0 and entry is not None:
                entry = None
                active = False
            elif previous != 0 and entry is None:
                entry = float(opens[i])
                entry_bar = i
                stop = entry - previous * params.atr_multiple * atr[i]

            if entry is None:
                if atr[i] > 0 and up[i]:
                    signal[i] = 1
                elif atr[i] > 0 and down[i]:
                    signal[i] = -1
                continue

            side = previous
            close = float(closes[i])
            if (side * (close - stop) <= 0
                    or i - entry_bar >= params.hold_bars):
                continue
            if side * (close / entry - 1) >= params.breakeven_pct:
                active = True
            if active:
                candidate = close - side * params.atr_multiple * atr[i]
                stop = max(stop, entry, candidate) if side > 0 else min(stop, entry, candidate)
            signal[i] = side

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
