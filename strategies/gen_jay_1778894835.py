from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumePlasticityParams:
    short_vol_window: int = 10
    long_vol_window: int = 60
    price_rank_window: int = 20
    plasticity_pct_threshold: float = 0.70
    price_min_pct: float = 0.35
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VolumePlasticityParams]):
    strategy_id = "gen_jay_1778894835"

    @classmethod
    def params_type(cls):
        return VolumePlasticityParams

    @staticmethod
    def warmup_bars(params: VolumePlasticityParams) -> int:
        # plasticity_rank rolls over long_vol_window bars of plasticity_score,
        # which itself needs long_vol_window bars; total: 2 * long_vol_window
        return params.long_vol_window * 2 + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolumePlasticityParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        vol = data["volume"].astype(float)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        short_rank = vol.rolling(
            params.short_vol_window, min_periods=params.short_vol_window
        ).rank(pct=True)
        long_rank = vol.rolling(
            params.long_vol_window, min_periods=params.long_vol_window
        ).rank(pct=True)

        # Positive when short-term volume percentile exceeds long-term baseline
        plasticity_score = short_rank - long_rank

        # Second-order percentile: how extreme is today's plasticity vs recent history
        ind["plasticity_rank"] = plasticity_score.rolling(
            params.long_vol_window, min_periods=params.long_vol_window
        ).rank(pct=True)

        roll_high = high.rolling(
            params.price_rank_window, min_periods=params.price_rank_window
        ).max()
        roll_low = low.rolling(
            params.price_rank_window, min_periods=params.price_rank_window
        ).min()
        range_ = (roll_high - roll_low).replace(0.0, np.nan)
        ind["price_position"] = (close - roll_low) / range_

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumePlasticityParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)

        plasticity_rank = indicators["plasticity_rank"].values
        price_position = indicators["price_position"].values

        entry_cond = (
            np.isfinite(plasticity_rank)
            & np.isfinite(price_position)
            & (plasticity_rank >= params.plasticity_pct_threshold)
            & (price_position >= params.price_min_pct)
        )

        in_trade = False
        bars_held = 0

        for i in range(n):
            if in_trade:
                bars_held += 1
                if bars_held > params.hold_bars:
                    in_trade = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            else:
                if entry_cond[i]:
                    raw_signal[i] = 1
                    in_trade = True
                    bars_held = 1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
