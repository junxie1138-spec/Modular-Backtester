from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityNormalizationParams:
    short_vol: int = 10
    long_vol: int = 63
    ma_period: int = 200
    atr_period: int = 14
    spike_ratio: float = 1.6
    normalize_ratio: float = 0.95
    refractory_days: int = 5
    recovery_window: int = 10
    target_vol: float = 0.12
    min_size: float = 0.25
    max_size: float = 1.0
    breakeven_pct: float = 0.04
    initial_stop_atr: float = 3.0
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[VolatilityNormalizationParams]):
    strategy_id = "gen_local_1791223483"

    @classmethod
    def params_type(cls) -> type[VolatilityNormalizationParams]:
        return VolatilityNormalizationParams

    @classmethod
    def warmup_bars(cls, params: VolatilityNormalizationParams) -> int:
        return max(params.ma_period, params.long_vol + 1)

    @classmethod
    def indicators(
        cls, data: pd.DataFrame, params: VolatilityNormalizationParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None)

        short_vol = (
            returns.rolling(params.short_vol, min_periods=params.short_vol)
            .std(ddof=0)
            .mul(np.sqrt(252.0))
            .replace([np.inf, -np.inf], np.nan)
        )
        long_vol = (
            returns.rolling(params.long_vol, min_periods=params.long_vol)
            .std(ddof=0)
            .mul(np.sqrt(252.0))
            .replace([np.inf, -np.inf], np.nan)
        )
        vol_ratio = short_vol.div(long_vol.replace(0.0, np.nan)).fillna(0.0)

        spike = vol_ratio.ge(params.spike_ratio)
        old_spike = (
            spike.astype(float)
            .shift(params.refractory_days)
            .rolling(params.recovery_window, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
        )
        fresh_spike = (
            spike.astype(float)
            .rolling(params.refractory_days, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
        )
        recovered = old_spike & ~fresh_spike & vol_ratio.le(params.normalize_ratio)
        entry_trigger = recovered & ~recovered.shift(1, fill_value=False)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                data["high"].sub(data["low"]),
                data["high"].sub(previous_close).abs(),
                data["low"].sub(previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=1).mean().fillna(0.0)
        ma200 = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        target_size = (
            params.target_vol
            / short_vol.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(params.max_size)

        return pd.DataFrame(
            {
                "short_vol": short_vol.fillna(0.0),
                "long_vol": long_vol.fillna(0.0),
                "vol_ratio": vol_ratio,
                "atr": atr,
                "ma200": ma200.fillna(np.inf),
                "entry_trigger": entry_trigger.fillna(False),
                "target_size": target_size.clip(params.min_size, params.max_size),
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityNormalizationParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        in_position = False
        entry_price = 0.0
        high_water = 0.0
        stop_price = 0.0
        held_bars = 0
        trailing_active = False

        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            high = float(data["high"].iloc[i])
            low = float(data["low"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            ma200 = float(indicators["ma200"].iloc[i])

            if not in_position:
                if bool(indicators["entry_trigger"].iloc[i]) and close > ma200:
                    in_position = True
                    entry_price = close
                    high_water = high
                    stop_price = entry_price - params.initial_stop_atr * atr
                    held_bars = 0
                    trailing_active = False
                    signal[i] = 1
                continue

            held_bars += 1
            high_water = max(high_water, high)

            if high_water >= entry_price * (1.0 + params.breakeven_pct):
                trailing_active = True

            if trailing_active:
                stop_price = max(
                    stop_price,
                    entry_price,
                    high_water - params.trail_atr * atr,
                )

            exit_now = (
                low <= stop_price
                or held_bars >= params.max_hold_bars
                or close <= ma200
            )
            if exit_now:
                in_position = False
                signal[i] = 0
            else:
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (
            indicators["target_size"]
            .shift(1)
            .fillna(params.min_size)
            .clip(params.min_size, params.max_size)
            .astype(float)
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
