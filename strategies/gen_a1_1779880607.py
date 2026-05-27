from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 9
    gap_z_threshold: float = 1.0
    range_ratio_max: float = 0.7
    profit_target_pct: float = 0.04
    time_stop_bars: int = 18
    refractory_bars: int = 5
    min_abs_gap: float = 0.002


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779880607"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.lookback, 2)) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = (open_ - prev_close) / prev_close
        bar_range = (high - low) / prev_close
        abs_gap = gap.abs()

        L = int(max(params.lookback, 2))
        median_abs_gap = abs_gap.rolling(L, min_periods=L).median()
        median_range = bar_range.rolling(L, min_periods=L).median()

        gap_z = abs_gap / median_abs_gap.replace(0.0, np.nan)
        range_z = bar_range / median_range.replace(0.0, np.nan)
        absorption = bar_range / abs_gap.replace(0.0, np.nan)

        return pd.DataFrame(
            {
                "gap": gap,
                "abs_gap": abs_gap,
                "bar_range": bar_range,
                "gap_z": gap_z,
                "range_z": range_z,
                "absorption": absorption,
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
        close = data["close"].to_numpy(dtype=np.float64)
        gap = indicators["gap"].to_numpy(dtype=np.float64)
        abs_gap = indicators["abs_gap"].to_numpy(dtype=np.float64)
        gap_z = indicators["gap_z"].to_numpy(dtype=np.float64)
        absorption = indicators["absorption"].to_numpy(dtype=np.float64)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        position = 0
        entry_price = 0.0
        bars_in_trade = 0
        cooldown = 0

        gap_thr = float(params.gap_z_threshold)
        abs_max = float(params.range_ratio_max)
        pt = float(params.profit_target_pct)
        time_stop = int(params.time_stop_bars)
        cool_reset = int(params.refractory_bars)
        min_gap = float(params.min_abs_gap)

        for i in range(n):
            if position != 0:
                bars_in_trade += 1
                px = close[i]
                if np.isfinite(px) and entry_price > 0.0:
                    if position == 1:
                        pnl = (px - entry_price) / entry_price
                    else:
                        pnl = (entry_price - px) / entry_price
                else:
                    pnl = 0.0

                exit_now = (pnl >= pt) or (bars_in_trade >= time_stop)

                if exit_now:
                    raw[i] = 0
                    position = 0
                    entry_price = 0.0
                    bars_in_trade = 0
                    cooldown = cool_reset
                else:
                    raw[i] = position
                continue

            if cooldown > 0:
                cooldown -= 1
                raw[i] = 0
                continue

            g = gap[i]
            ag = abs_gap[i]
            gz = gap_z[i]
            ab = absorption[i]
            px = close[i]

            if not (
                np.isfinite(g)
                and np.isfinite(ag)
                and np.isfinite(gz)
                and np.isfinite(ab)
                and np.isfinite(px)
            ):
                raw[i] = 0
                continue

            big_gap = (gz >= gap_thr) and (ag >= min_gap)
            absorbed = ab <= abs_max

            if big_gap and absorbed:
                if g < 0.0:
                    position = 1
                    raw[i] = 1
                else:
                    position = -1
                    raw[i] = -1
                entry_price = px
                bars_in_trade = 0
            else:
                raw[i] = 0

        df = pd.DataFrame({"signal": raw, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
