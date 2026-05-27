from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_lookback: int = 60
    infected_window: int = 20
    saturation_threshold: float = 0.85
    dd_zscore_window: int = 100
    dd_zscore_entry: float = -1.2
    peak_window: int = 5
    regime_ma: int = 200
    atr_window: int = 14
    atr_stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779650246"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(
            params.regime_ma,
            params.dd_zscore_window + params.dd_lookback,
            params.atr_window + 1,
        ) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        roll_max = close.rolling(params.dd_lookback, min_periods=1).max()
        dd_long = close / roll_max - 1.0
        in_dd_long = (close < roll_max).astype(float)

        roll_min = close.rolling(params.dd_lookback, min_periods=1).min()
        dd_short = close / roll_min - 1.0
        in_dd_short = (close > roll_min).astype(float)

        infected_long = in_dd_long.rolling(
            params.infected_window, min_periods=params.infected_window
        ).mean()
        infected_short = in_dd_short.rolling(
            params.infected_window, min_periods=params.infected_window
        ).mean()

        dd_long_mean = dd_long.rolling(
            params.dd_zscore_window, min_periods=params.dd_zscore_window
        ).mean()
        dd_long_std = dd_long.rolling(
            params.dd_zscore_window, min_periods=params.dd_zscore_window
        ).std(ddof=0)
        dd_long_z = (dd_long - dd_long_mean) / dd_long_std.replace(0.0, np.nan)

        dd_short_mean = dd_short.rolling(
            params.dd_zscore_window, min_periods=params.dd_zscore_window
        ).mean()
        dd_short_std = dd_short.rolling(
            params.dd_zscore_window, min_periods=params.dd_zscore_window
        ).std(ddof=0)
        dd_short_z = (dd_short - dd_short_mean) / dd_short_std.replace(0.0, np.nan)

        inf_long_rmax = infected_long.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        inf_long_peaked = (inf_long_rmax - infected_long) > 0

        inf_short_rmax = infected_short.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        inf_short_peaked = (inf_short_rmax - infected_short) > 0

        ma = close.rolling(params.regime_ma, min_periods=params.regime_ma).mean()
        bull_regime = close > ma
        bear_regime = close < ma

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["dd_long"] = dd_long
        out["dd_short"] = dd_short
        out["dd_long_z"] = dd_long_z
        out["dd_short_z"] = dd_short_z
        out["infected_long"] = infected_long
        out["infected_short"] = infected_short
        out["inf_long_peaked"] = inf_long_peaked.astype(float)
        out["inf_short_peaked"] = inf_short_peaked.astype(float)
        out["bull_regime"] = bull_regime.astype(float)
        out["bear_regime"] = bear_regime.astype(float)
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        infected_long = indicators["infected_long"].to_numpy(dtype=float)
        infected_short = indicators["infected_short"].to_numpy(dtype=float)
        dd_long_z = indicators["dd_long_z"].to_numpy(dtype=float)
        dd_short_z = indicators["dd_short_z"].to_numpy(dtype=float)
        inf_long_peaked = indicators["inf_long_peaked"].to_numpy(dtype=float)
        inf_short_peaked = indicators["inf_short_peaked"].to_numpy(dtype=float)
        bull = indicators["bull_regime"].to_numpy(dtype=float)
        bear = indicators["bear_regime"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        long_entry = (
            (infected_long >= params.saturation_threshold)
            & (dd_long_z <= params.dd_zscore_entry)
            & (inf_long_peaked > 0.5)
            & (bull > 0.5)
        )
        short_entry = (
            (infected_short >= params.saturation_threshold)
            & (dd_short_z >= -params.dd_zscore_entry)
            & (inf_short_peaked > 0.5)
            & (bear > 0.5)
        )

        position = 0
        entry_price = 0.0
        entry_atr = 0.0

        for i in range(n):
            if position == 0:
                if bool(long_entry[i]) and not np.isnan(atr[i]):
                    position = 1
                    entry_price = float(close[i])
                    entry_atr = float(atr[i])
                elif bool(short_entry[i]) and not np.isnan(atr[i]):
                    position = -1
                    entry_price = float(close[i])
                    entry_atr = float(atr[i])
            else:
                if position == 1:
                    stop = entry_price - params.atr_stop_mult * entry_atr
                    if close[i] <= stop:
                        position = 0
                        entry_price = 0.0
                        entry_atr = 0.0
                else:
                    stop = entry_price + params.atr_stop_mult * entry_atr
                    if close[i] >= stop:
                        position = 0
                        entry_price = 0.0
                        entry_atr = 0.0

            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
