from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalComplexityParams:
    pattern_window: int = 12
    complexity_lag: int = 3
    complexity_jump: float = 0.08
    min_complexity: float = 0.45
    infection_fast: int = 4
    infection_slow: int = 20
    infection_gap: float = 0.15
    atr_length: int = 14
    trail_atr: float = 2.0
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[OrdinalComplexityParams]):
    strategy_id = "gen_local_1791223309"

    @classmethod
    def params_type(cls) -> type[OrdinalComplexityParams]:
        return OrdinalComplexityParams

    @classmethod
    def warmup_bars(cls, params: OrdinalComplexityParams) -> int:
        return int(max(
            params.pattern_window + params.complexity_lag + 3,
            params.infection_slow + 1,
            params.atr_length + 1,
        ))

    def indicators(
        self, data: pd.DataFrame, params: OrdinalComplexityParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        raw_ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        ret = raw_ret.fillna(0.0)

        a = raw_ret.shift(2)
        b = raw_ret.shift(1)
        c = raw_ret
        valid = a.notna() & b.notna() & c.notna()

        rank_a = (a > b).astype(int) + (a > c).astype(int)
        rank_b = (b >= a).astype(int) + (b > c).astype(int)
        pattern = rank_a * 3 + rank_b

        entropy = pd.Series(0.0, index=data.index)
        for code in (1, 2, 3, 5, 6, 7):
            probability = (
                ((pattern == code) & valid)
                .astype(float)
                .rolling(params.pattern_window, min_periods=params.pattern_window)
                .sum()
                / params.pattern_window
            )
            entropy -= probability.where(probability > 0.0, 1.0) * np.log(
                probability.where(probability > 0.0, 1.0)
            )
        entropy = (entropy / np.log(6.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        positive = (ret > 0.0).astype(float)
        infection_fast = positive.rolling(
            params.infection_fast, min_periods=params.infection_fast
        ).mean().fillna(0.0)
        infection_slow = positive.rolling(
            params.infection_slow, min_periods=params.infection_slow
        ).mean().fillna(0.0)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"].astype(float) - data["low"].astype(float),
                (data["high"].astype(float) - previous_close).abs(),
                (data["low"].astype(float) - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(
            params.atr_length, min_periods=params.atr_length
        ).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        momentum = close.pct_change(
            params.infection_fast, fill_method=None
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ready = pd.Series(
            np.arange(len(data)) >= self.warmup_bars(params), index=data.index
        )

        return pd.DataFrame(
            {
                "ordinal_complexity": entropy,
                "complexity_change": entropy.diff(params.complexity_lag).fillna(0.0),
                "infection_fast": infection_fast,
                "infection_slow": infection_slow,
                "momentum": momentum,
                "atr": atr,
                "ready": ready.astype(bool),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalComplexityParams,
    ) -> SignalFrame:
        complexity_agrees = (
            (indicators["ordinal_complexity"] >= params.min_complexity)
            & (indicators["complexity_change"] >= params.complexity_jump)
        )
        contagion_agrees = (
            indicators["infection_fast"]
            >= indicators["infection_slow"] + params.infection_gap
        ) & (indicators["momentum"] > 0.0)
        entries = (
            complexity_agrees & contagion_agrees & indicators["ready"]
        ).fillna(False).to_numpy(dtype=bool)

        closes = data["close"].astype(float).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        in_trade = False
        high_water = 0.0
        held = 0

        for i in range(len(data)):
            if not in_trade:
                if entries[i]:
                    in_trade = True
                    high_water = closes[i]
                    held = 0
                    signal[i] = 1
                continue

            held += 1
            high_water = max(high_water, closes[i])
            trailing_stop = high_water - params.trail_atr * atr[i]
            if closes[i] < trailing_stop or held >= params.max_hold:
                in_trade = False
                signal[i] = 0
            else:
                signal[i] = 1

        df = pd.DataFrame(
            {"signal": signal, "size": np.ones(len(data), dtype=float)},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
