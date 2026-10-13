from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeeklyEchoParams:
    rank_window: int = 63
    correlation_window: int = 42
    history_window: int = 126
    echo_percentile: float = 0.80
    minimum_correlation: float = 0.10
    correlation_advantage: float = 0.10
    pullback_percentile: float = 0.40
    reference_percentile: float = 0.65
    atr_window: int = 14
    atr_multiple: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[WeeklyEchoParams]):
    strategy_id = "gen_local-w2_1790992999"

    @classmethod
    def params_type(cls) -> type[WeeklyEchoParams]:
        return WeeklyEchoParams

    def warmup_bars(self, params: WeeklyEchoParams) -> int:
        return int(max(
            200,
            params.rank_window + 5 + params.correlation_window
            + params.history_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: WeeklyEchoParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        out["return"] = close.div(previous_close.where(previous_close > 0)).sub(1)
        out["return_rank"] = out["return"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        rank = out["return_rank"]
        out["weekly_correlation"] = rank.rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(rank.shift(5))
        out["daily_correlation"] = rank.rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(rank.shift(1))
        out["echo_rank"] = out["weekly_correlation"].rolling(
            params.history_window, min_periods=params.history_window
        ).rank(pct=True)
        out["next_weekly_reference"] = rank.shift(4)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"].sub(data["low"]).abs(),
            data["high"].sub(previous_close).abs(),
            data["low"].sub(previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WeeklyEchoParams,
    ) -> SignalFrame:
        close = data["close"].astype(float)
        required = [
            "return", "return_rank", "weekly_correlation",
            "daily_correlation", "echo_rank", "next_weekly_reference",
            "ma200", "atr",
        ]
        ready = indicators[required].notna().all(axis=1)
        ready &= close.notna() & np.isfinite(close) & (close > 0)
        ready &= indicators["atr"] > 0
        entry = (
            ready
            & (close > indicators["ma200"])
            & (indicators["weekly_correlation"] >= params.minimum_correlation)
            & (indicators["weekly_correlation"] >=
               indicators["daily_correlation"] + params.correlation_advantage)
            & (indicators["echo_rank"] >= params.echo_percentile)
            & (indicators["return"] < 0)
            & (indicators["return_rank"] <= params.pullback_percentile)
            & (indicators["next_weekly_reference"] >= params.reference_percentile)
        ).fillna(False)
        closes = close.to_numpy()
        atrs = indicators["atr"].to_numpy()
        averages = indicators["ma200"].to_numpy()
        valid = ready.to_numpy(dtype=bool)
        entries = entry.to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        held_bars = 0
        for i in range(len(data)):
            if active:
                held_bars += 1
                if not valid[i]:
                    active = False
                else:
                    # The first close included is the next-bar fill day's close.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_multiple * atrs[i])
                    if (
                        closes[i] <= stop
                        or closes[i] <= averages[i]
                        or held_bars >= params.holding_bars
                    ):
                        active = False
            elif entries[i]:
                active = True
                high_water = -np.inf
                stop = -np.inf
                held_bars = 0
            signal[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
