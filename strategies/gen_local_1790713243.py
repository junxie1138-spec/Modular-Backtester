from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    momentum_atr_min: float = 0.8
    volume_ratio_min: float = 1.3
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713243"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return 21

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        volume_mean = data["volume"].rolling(20, min_periods=20).mean()
        return pd.DataFrame(
            {
                "atr": atr,
                "momentum": data["close"].pct_change(3),
                "volume_ratio": data["volume"] / volume_mean,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        momentum = indicators["momentum"].to_numpy(dtype=float)
        volume_ratio = indicators["volume_ratio"].to_numpy(dtype=float)
        raw_signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.01, dtype=float)
        side = 0
        entry_price = 0.0
        entry_atr = 0.0
        held = 0

        for i in range(len(data)):
            valid = (
                np.isfinite(atr[i])
                and atr[i] > 0
                and np.isfinite(momentum[i])
                and np.isfinite(volume_ratio[i])
            )
            if side:
                held += 1
                stopped = close[i] <= entry_price - params.stop_atr * entry_atr if side > 0 else close[i] >= entry_price + params.stop_atr * entry_atr
                if stopped or held >= 4:
                    side = 0
                else:
                    raw_signal[i] = side
                continue

            if not valid or volume_ratio[i] < params.volume_ratio_min:
                continue
            move_atr = abs(momentum[i]) * close[i] / atr[i]
            if move_atr < params.momentum_atr_min:
                continue
            side = 1 if momentum[i] > 0 else -1
            entry_price = close[i]
            entry_atr = atr[i]
            held = 0
            raw_signal[i] = side
            move_strength = min(move_atr / (2.0 * params.momentum_atr_min), 1.0)
            volume_strength = min((volume_ratio[i] - 1.0) / (params.volume_ratio_min - 1.0), 1.0)
            size[i] = max(0.01, move_strength * volume_strength)

        df = pd.DataFrame({"signal": raw_signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
