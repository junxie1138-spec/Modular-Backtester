from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenParams:
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[GenParams]):
    strategy_id = "gen_a1_1779877882"

    @classmethod
    def params_type(cls) -> type[GenParams]:
        return GenParams

    @classmethod
    def warmup_bars(cls, params: GenParams) -> int:
        # corridor(20) -> dispersion(10) -> percentile(60) chain ~ 90 bars
        return 100

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GenParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        corridor_n = 20
        disp_n = 10
        pct_n = 60
        atr_n = 14

        corr_high = close.rolling(corridor_n, min_periods=corridor_n).max()
        corr_low = close.rolling(corridor_n, min_periods=corridor_n).min()
        corr_width = (corr_high - corr_low).replace(0.0, np.nan)
        rel = ((close - corr_low) / corr_width).clip(0.0, 1.0)

        disp = rel.rolling(disp_n, min_periods=disp_n).std()
        mean_rel = rel.rolling(disp_n, min_periods=disp_n).mean()
        disp_floor = disp.rolling(pct_n, min_periods=pct_n).quantile(0.20)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_n, min_periods=atr_n).mean()

        out = pd.DataFrame(
            {
                "rel": rel,
                "disp": disp,
                "disp_floor": disp_floor,
                "mean_rel": mean_rel,
                "atr": atr,
            },
            index=data.index,
        )
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        disp = indicators["disp"].to_numpy(dtype=float)
        disp_floor = indicators["disp_floor"].to_numpy(dtype=float)
        mean_rel = indicators["mean_rel"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        be_pct = float(params.breakeven_pct)
        k_atr = float(params.trail_atr_mult)
        mean_rel_floor = 0.35
        init_stop_mult = 2.0

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        peak_price = 0.0
        be_armed = False

        for i in range(n):
            c = close[i]
            a = atr[i]
            d = disp[i]
            df_floor = disp_floor[i]
            mr = mean_rel[i]

            inputs_ready = (
                not np.isnan(d)
                and not np.isnan(df_floor)
                and not np.isnan(mr)
                and not np.isnan(a)
                and a > 0.0
            )

            if not in_trade:
                if inputs_ready and d < df_floor and mr < mean_rel_floor:
                    in_trade = True
                    entry_price = c
                    peak_price = c
                    be_armed = False
                    stop_price = entry_price - init_stop_mult * a
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                if c > peak_price:
                    peak_price = c
                if (not be_armed) and c >= entry_price * (1.0 + be_pct):
                    be_armed = True
                    if entry_price > stop_price:
                        stop_price = entry_price
                if be_armed and (not np.isnan(a)):
                    candidate = peak_price - k_atr * a
                    if candidate > stop_price:
                        stop_price = candidate
                if c <= stop_price:
                    in_trade = False
                    entry_price = 0.0
                    stop_price = 0.0
                    peak_price = 0.0
                    be_armed = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        signal = pd.Series(raw, index=data.index, name="signal")
        signal = signal.shift(1).fillna(0).astype(int)

        size = pd.Series(1.0, index=data.index, name="size", dtype=float)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
