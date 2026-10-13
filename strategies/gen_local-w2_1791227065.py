from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    frac_d: float = 0.45
    frac_window: int = 24
    fast_span: int = 3
    slow_span: int = 15
    norm_window: int = 63
    shift_threshold: float = 1.0
    arm_pct: float = 0.012
    trail_atr: float = 1.7
    atr_window: int = 14
    max_hold: int = 5
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791227065"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.frac_window + params.norm_window + params.slow_span

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        log_close = np.log(data["close"].where(data["close"] > 0)).ffill()
        weights = [1.0]
        for k in range(1, params.frac_window):
            weights.append(-weights[-1] * (params.frac_d - k + 1) / k)
        fd = sum(
            (weight * log_close.shift(k) for k, weight in enumerate(weights)),
            pd.Series(0.0, index=data.index),
        )
        spread = fd.ewm(span=params.fast_span, adjust=False).mean() - fd.ewm(
            span=params.slow_span, adjust=False
        ).mean()
        baseline = spread.rolling(params.norm_window).mean()
        scale = spread.rolling(params.norm_window).std().replace(0, np.nan)
        z = ((spread - baseline) / scale).replace([np.inf, -np.inf], np.nan)
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame(
            {
                "fd": fd.fillna(0.0),
                "spread": spread.fillna(0.0),
                "baseline": baseline.fillna(0.0),
                "scale": scale.fillna(0.0),
                "z": z.fillna(0.0),
                "atr": atr.fillna(0.0),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        raw = pd.Series(0, index=data.index, dtype=int)
        sizes = pd.Series(float(params.min_size), index=data.index)
        z = indicators["z"]
        atr = indicators["atr"]
        position = 0
        entry_price = None
        stop = None
        held = 0
        size = float(params.min_size)

        for i in range(len(data)):
            strength = float(z.iloc[i])
            previous = float(z.iloc[i - 1]) if i else 0.0
            if position == 0:
                direction = 0
                if strength >= params.shift_threshold and previous < params.shift_threshold:
                    direction = 1
                elif strength <= -params.shift_threshold and previous > -params.shift_threshold:
                    direction = -1
                if direction:
                    position = direction
                    entry_price = None
                    stop = None
                    held = 0
                    excess = (abs(strength) - params.shift_threshold) / (2 * params.shift_threshold)
                    size = float(np.clip(
                        params.min_size + (params.max_size - params.min_size) * excess,
                        params.min_size,
                        params.max_size,
                    ))
                    raw.iloc[i] = position
                    sizes.iloc[i] = size
                continue

            if entry_price is None:
                entry_price = float(data["open"].iloc[i])
                stop = entry_price - position * params.trail_atr * float(atr.iloc[i - 1])
            held += 1
            high = float(data["high"].iloc[i])
            low = float(data["low"].iloc[i])
            close = float(data["close"].iloc[i])
            hit_stop = low <= stop if position == 1 else high >= stop
            if hit_stop or held >= params.max_hold:
                position = 0
                entry_price = None
                stop = None
                continue

            if position == 1 and high >= entry_price * (1 + params.arm_pct):
                stop = max(stop, entry_price, close - params.trail_atr * float(atr.iloc[i]))
            elif position == -1 and low <= entry_price * (1 - params.arm_pct):
                stop = min(stop, entry_price, close + params.trail_atr * float(atr.iloc[i]))
            raw.iloc[i] = position
            sizes.iloc[i] = size

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = sizes.shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
