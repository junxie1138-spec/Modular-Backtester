from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEpidemicParams:
    gap_lookback: int = 20
    channel_lookback: int = 15
    curve_lag: int = 3
    gap_z_threshold: float = 1.5
    infection_floor: float = 0.15
    infection_ceiling: float = 0.55
    profit_target: float = 0.025
    max_hold_bars: int = 4
    min_size: float = 0.35


class GeneratedStrategy(BaseStrategy[GapEpidemicParams]):
    strategy_id = "gen_local_1791223600"

    @classmethod
    def params_type(cls) -> type[GapEpidemicParams]:
        return GapEpidemicParams

    @classmethod
    def warmup_bars(cls, params: GapEpidemicParams) -> int:
        return max(
            params.gap_lookback + params.curve_lag + 1,
            params.channel_lookback + 1,
        )

    @classmethod
    def indicators(
        cls, data: pd.DataFrame, params: GapEpidemicParams
    ) -> pd.DataFrame:
        prior_close = data["close"].shift(1)
        gap = data["open"].div(prior_close).sub(1.0)
        gap = gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        gap_mean = (
            gap.rolling(params.gap_lookback, min_periods=params.gap_lookback)
            .mean()
            .shift(1)
        )
        gap_std = (
            gap.rolling(params.gap_lookback, min_periods=params.gap_lookback)
            .std(ddof=0)
            .shift(1)
            .replace(0.0, np.nan)
        )
        gap_z = gap.sub(gap_mean).div(gap_std)

        positive_gap = gap.gt(0.0).astype(float)
        infected = (
            positive_gap.rolling(
                params.gap_lookback, min_periods=params.gap_lookback
            )
            .mean()
            .shift(1)
        )
        epidemic_curve = infected.mul(1.0 - infected)
        curve_gain = epidemic_curve.sub(epidemic_curve.shift(params.curve_lag))
        prior_high = (
            data["high"]
            .rolling(
                params.channel_lookback, min_periods=params.channel_lookback
            )
            .max()
            .shift(1)
        )

        return pd.DataFrame(
            {
                "gap": gap,
                "gap_z": gap_z.fillna(0.0),
                "infected": infected.fillna(0.0),
                "epidemic_curve": epidemic_curve.fillna(0.0),
                "curve_gain": curve_gain.fillna(0.0),
                "prior_high": prior_high.fillna(data["close"]),
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEpidemicParams,
    ) -> SignalFrame:
        entry = (
            indicators["gap_z"].ge(params.gap_z_threshold)
            & indicators["infected"].between(
                params.infection_floor, params.infection_ceiling
            )
            & indicators["curve_gain"].gt(0.0)
            & data["close"].gt(indicators["prior_high"])
            & data["close"].gt(data["open"])
        ).fillna(False)

        gap_score = (
            indicators["gap_z"].sub(params.gap_z_threshold).div(2.0).clip(0.0, 1.0)
        )
        curve_score = indicators["curve_gain"].div(0.10).clip(0.0, 1.0)
        conviction = gap_score.mul(0.65).add(curve_score.mul(0.35)).fillna(0.0)
        proposed_size = params.min_size + (1.0 - params.min_size) * conviction

        signal = pd.Series(0, index=data.index, dtype=int)
        size = pd.Series(params.min_size, index=data.index, dtype=float)
        in_position = False
        entry_price = 0.0
        entry_size = params.min_size
        bars_held = 0

        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if in_position:
                bars_held += 1
                if (
                    close >= entry_price * (1.0 + params.profit_target)
                    or bars_held >= params.max_hold_bars
                ):
                    in_position = False
                else:
                    signal.iloc[i] = 1
                    size.iloc[i] = entry_size
            elif bool(entry.iloc[i]):
                in_position = True
                entry_price = close
                entry_size = float(proposed_size.iloc[i])
                bars_held = 0
                signal.iloc[i] = 1
                size.iloc[i] = entry_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
