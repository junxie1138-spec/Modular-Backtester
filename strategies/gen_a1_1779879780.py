from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA11779879780Params:
    trend_window: int = 50
    snr_window: int = 20
    rank_window: int = 252
    rank_threshold: float = 0.85
    hold_bars: int = 2
    regime_ma: int = 200


class GeneratedStrategy(BaseStrategy[GenA11779879780Params]):
    strategy_id = "gen_a1_1779879780"

    @classmethod
    def params_type(cls) -> type[GenA11779879780Params]:
        return GenA11779879780Params

    def warmup_bars(self, params: GenA11779879780Params) -> int:
        return (
            max(
                params.trend_window,
                params.snr_window,
                params.rank_window,
                params.regime_ma,
            )
            + 2
        )

    def indicators(
        self, data: pd.DataFrame, params: GenA11779879780Params
    ) -> pd.DataFrame:
        close = data["close"].astype(float)

        # --- Primitive 1: trend z-score percentile rank ---
        # How many std-units is price above its trend MA? Rank that within rank_window.
        sma_trend = close.rolling(
            params.trend_window, min_periods=params.trend_window
        ).mean()
        std_trend = close.rolling(
            params.trend_window, min_periods=params.trend_window
        ).std(ddof=0)
        trend_z = (close - sma_trend) / std_trend.replace(0.0, np.nan)
        trend_z_rank = trend_z.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        # --- Primitive 2: efficiency ratio (signal-to-noise) percentile rank ---
        # |net move over N bars| / sum of |bar-to-bar moves| over N bars.
        # High = clean directional move; low = choppy.
        net_move = (close - close.shift(params.snr_window)).abs()
        abs_moves = (
            close.diff()
            .abs()
            .rolling(params.snr_window, min_periods=params.snr_window)
            .sum()
        )
        snr = net_move / abs_moves.replace(0.0, np.nan)
        snr_rank = snr.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        # Efficiency ratio is unsigned: require the net move is actually up.
        direction_up = (close > close.shift(params.snr_window)).astype(float)

        # Long-only regime filter: price above long MA.
        sma_regime = close.rolling(
            params.regime_ma, min_periods=params.regime_ma
        ).mean()
        above_regime = (close > sma_regime).astype(float)

        return pd.DataFrame(
            {
                "trend_z_rank": trend_z_rank,
                "snr_rank": snr_rank,
                "direction_up": direction_up,
                "above_regime": above_regime,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA11779879780Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=np.int64)

        trend_rank = indicators["trend_z_rank"].fillna(0.0).to_numpy()
        snr_rank = indicators["snr_rank"].fillna(0.0).to_numpy()
        dir_up = indicators["direction_up"].fillna(0.0).to_numpy()
        above_regime = indicators["above_regime"].fillna(0.0).to_numpy()

        threshold = float(params.rank_threshold)
        hold = int(params.hold_bars)
        if hold < 1:
            hold = 1

        # Fixed-bar exit: once we enter, we hold exactly `hold` bars of signal=1
        # and ignore further entry conditions until the hold elapses. Path-
        # dependent counter is intentionally a small Python loop.
        bars_remaining = 0
        for i in range(n):
            if bars_remaining > 0:
                signal[i] = 1
                bars_remaining -= 1
                continue
            # Two-primitive AND: both percentile ranks must be at the extreme.
            both_extreme = (
                trend_rank[i] >= threshold and snr_rank[i] >= threshold
            )
            up_gates = dir_up[i] >= 1.0 and above_regime[i] >= 1.0
            if both_extreme and up_gates:
                signal[i] = 1
                bars_remaining = hold - 1

        df = pd.DataFrame({"signal": signal}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(
            data=df, signal_column="signal", size_column="size"
        )
