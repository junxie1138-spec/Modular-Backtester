from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeVolatilityParams:
    fast_window: int = 18
    slow_window: int = 63
    pressure_window: int = 18
    regime_on: float = 1.15
    regime_off: float = 0.90
    pressure_threshold: float = 0.08
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[RegimeVolatilityParams]):
    strategy_id = "gen_local-w2_1791223051"

    @classmethod
    def params_type(cls) -> type[RegimeVolatilityParams]:
        return RegimeVolatilityParams

    @classmethod
    def warmup_bars(cls, params: RegimeVolatilityParams) -> int:
        return max(params.fast_window, params.slow_window, params.pressure_window) + 1

    @classmethod
    def indicators(
        cls, data: pd.DataFrame, params: RegimeVolatilityParams
    ) -> pd.DataFrame:
        returns = data["close"].pct_change()
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        fast_vol = returns.rolling(params.fast_window).std()
        slow_vol = returns.rolling(params.slow_window).std()
        vol_ratio = fast_vol.div(slow_vol.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        signed_range = true_range * np.sign(returns.fillna(0.0))
        range_sum = true_range.rolling(params.pressure_window).sum()
        pressure = signed_range.rolling(params.pressure_window).sum().div(
            range_sum.replace(0.0, np.nan)
        ).fillna(0.0)
        momentum = data["close"].pct_change(params.fast_window).fillna(0.0)

        high_vol = np.zeros(len(data), dtype=bool)
        active = False
        for i, ratio in enumerate(vol_ratio.to_numpy(dtype=float)):
            if active and ratio < params.regime_off:
                active = False
            elif not active and ratio > params.regime_on:
                active = True
            high_vol[i] = active

        return pd.DataFrame(
            {
                "vol_ratio": vol_ratio,
                "high_vol": high_vol,
                "momentum": momentum,
                "range_pressure": pressure,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RegimeVolatilityParams,
    ) -> SignalFrame:
        up_agreement = (indicators["momentum"] > 0.0) & (
            indicators["range_pressure"] > params.pressure_threshold
        )
        down_agreement = (indicators["momentum"] < 0.0) & (
            indicators["range_pressure"] < -params.pressure_threshold
        )
        high_vol = indicators["high_vol"].astype(bool)

        long_entry = (high_vol & up_agreement) | (~high_vol & down_agreement)
        short_entry = (high_vol & down_agreement) | (~high_vol & up_agreement)

        position = 0
        states = np.zeros(len(data), dtype=int)
        for i, (go_long, go_short) in enumerate(
            zip(long_entry.to_numpy(dtype=bool), short_entry.to_numpy(dtype=bool))
        ):
            if position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            elif position == 0:
                if go_long:
                    position = 1
                elif go_short:
                    position = -1
            states[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(states, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(max(params.size, 0.01))
        return SignalFrame(data=df, signal_column="signal", size_column="size")
