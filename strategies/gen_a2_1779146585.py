from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 60
    ma_window: int = 200
    wave_window: int = 3
    min_underwater: int = 5
    dd_threshold: float = 0.04
    profit_target: float = 0.05
    time_stop: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779146585"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.peak_window, params.ma_window, params.wave_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        ret = close.pct_change()
        peak = close.rolling(params.peak_window, min_periods=1).max()
        drawdown = close / peak - 1.0

        underwater = (drawdown < 0.0).astype(int)
        reset_grp = (underwater == 0).cumsum()
        underwater_bars = underwater.groupby(reset_grp).cumsum()

        wave = ret.rolling(params.wave_window).sum()
        ma = close.rolling(params.ma_window).mean()

        ind["drawdown"] = drawdown
        ind["underwater_bars"] = underwater_bars.astype(float)
        ind["wave"] = wave
        ind["wave_prev"] = wave.shift(1)
        ind["ma"] = ma
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        uw = indicators["underwater_bars"].to_numpy(dtype=float)
        wave = indicators["wave"].to_numpy(dtype=float)
        wave_prev = indicators["wave_prev"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        bars_held = 0
        pt = float(params.profit_target)
        ts = int(params.time_stop)
        min_uw = int(params.min_underwater)
        dd_thr = float(params.dd_threshold)

        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= ts:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1
                continue

            valid = (
                not np.isnan(dd[i])
                and not np.isnan(uw[i])
                and not np.isnan(wave[i])
                and not np.isnan(wave_prev[i])
                and not np.isnan(ma[i])
            )
            if not valid:
                continue

            entry = (
                uw[i] >= min_uw
                and dd[i] <= -dd_thr
                and wave[i] > 0.0
                and wave_prev[i] <= 0.0
                and close[i] > ma[i]
            )
            if entry:
                in_pos = True
                entry_price = close[i]
                bars_held = 0
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
