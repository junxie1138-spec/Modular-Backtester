from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_window: int = 10
    long_window: int = 40
    rank_window: int = 252
    atr_window: int = 14
    atr_mult: float = 2.5
    max_holding_bars: int = 10
    upper_rank_threshold: float = 0.70
    lower_rank_threshold: float = 0.30
    confirm_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779647283"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(params.rank_window + params.long_window + 5)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        up_bar = (close.diff() > 0).astype(float)

        short_density = up_bar.rolling(
            params.short_window, min_periods=params.short_window
        ).mean()
        long_density = up_bar.rolling(
            params.long_window, min_periods=params.long_window
        ).mean()

        denom = long_density.where(long_density > 0.0)
        r_number = short_density / denom

        r_rank = r_number.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return pd.DataFrame(
            {
                "short_density": short_density,
                "long_density": long_density,
                "r_number": r_number,
                "r_rank": r_rank,
                "atr": atr,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        r_rank = indicators["r_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        upper = float(params.upper_rank_threshold)
        lower = float(params.lower_rank_threshold)
        confirm = max(int(params.confirm_bars), 1)
        k = float(params.atr_mult)
        max_hold = max(int(params.max_holding_bars), 1)

        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            rk = r_rank[i]
            px = close[i]
            a = atr[i]

            if in_position:
                bars_held += 1
                stop_level = entry_price - k * entry_atr
                if not np.isnan(px) and px <= stop_level:
                    signal[i] = 0
                    in_position = False
                    bars_held = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
                if not np.isnan(rk) and rk < lower:
                    signal[i] = 0
                    in_position = False
                    bars_held = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
                if bars_held >= max_hold:
                    signal[i] = 0
                    in_position = False
                    bars_held = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
                signal[i] = 1
            else:
                if i >= confirm - 1 and not np.isnan(a) and a > 0.0 and not np.isnan(px):
                    ok = True
                    for j in range(confirm):
                        v = r_rank[i - j]
                        if np.isnan(v) or v <= upper:
                            ok = False
                            break
                    if ok:
                        signal[i] = 1
                        in_position = True
                        entry_price = px
                        entry_atr = a
                        bars_held = 0

        out = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        out["signal"] = out["signal"].shift(1).fillna(0).astype(int)
        out["size"] = out["size"].astype(float)

        return SignalFrame(data=out, signal_column="signal", size_column="size")
