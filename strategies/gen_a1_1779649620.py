from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_threshold: float = 0.001
    cascade_count: int = 3
    early_window: int = 7
    ma_period: int = 200
    atr_period: int = 14
    atr_stop_mult: float = 1.75
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779649620"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.ma_period, params.atr_period)) + 2

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prev_close = data["close"].shift(1)

        gap_pct = (data["open"] - prev_close) / prev_close
        out["gap_pct"] = gap_pct
        gap_up = (gap_pct > params.gap_threshold).fillna(False)
        out["gap_up"] = gap_up.astype("int64")

        ma = data["close"].rolling(params.ma_period, min_periods=params.ma_period).mean()
        out["ma"] = ma
        regime_up = (data["close"] > ma).fillna(False)
        out["regime_up"] = regime_up.astype("int64")

        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close).abs()
        tr3 = (data["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        years = np.asarray(data.index.year, dtype=np.int64)
        months = np.asarray(data.index.month, dtype=np.int64)
        month_id_arr = years * 100 + months
        month_id = pd.Series(month_id_arr, index=data.index)
        out["month_id"] = month_id

        in_month_bar = month_id.groupby(month_id).cumcount() + 1
        out["in_month_bar"] = in_month_bar.astype("int64")

        cum_gap_ups = out["gap_up"].groupby(month_id).cumsum()
        out["cum_gap_ups_month"] = cum_gap_ups.astype("int64")

        trigger = (
            (out["gap_up"] == 1)
            & (out["cum_gap_ups_month"] == int(params.cascade_count))
            & (out["in_month_bar"] <= int(params.early_window))
            & (out["regime_up"] == 1)
            & out["atr"].notna()
        )
        out["trigger"] = trigger.astype("int64")

        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=np.int64)

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        trigger = indicators["trigger"].to_numpy()

        in_position = False
        high_water = 0.0
        bars_held = 0
        stop_mult = float(params.atr_stop_mult)
        max_hold = int(params.max_hold_bars)

        for i in range(n):
            if not in_position:
                if (
                    trigger[i] == 1
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                    and np.isfinite(close[i])
                ):
                    in_position = True
                    signals[i] = 1
                    high_water = float(close[i])
                    bars_held = 1
            else:
                bars_held += 1
                if np.isfinite(close[i]) and close[i] > high_water:
                    high_water = float(close[i])

                stop_hit = False
                if (
                    np.isfinite(atr[i])
                    and np.isfinite(close[i])
                    and close[i] < (high_water - stop_mult * atr[i])
                ):
                    stop_hit = True

                time_stop = bars_held >= max_hold

                if stop_hit or time_stop:
                    in_position = False
                    signals[i] = 0
                    high_water = 0.0
                    bars_held = 0
                else:
                    signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
