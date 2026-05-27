from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeekdaySpringParams:
    ma_window: int = 200
    seasonal_min_obs: int = 60
    tension_window: int = 10
    tension_threshold: float = -0.015
    profit_target_pct: float = 0.012
    time_stop_bars: int = 2
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[WeekdaySpringParams]):
    strategy_id = "gen_a1_1779651432"

    @classmethod
    def params_type(cls):
        return WeekdaySpringParams

    def warmup_bars(self, params):
        return int(max(params.ma_window, params.seasonal_min_obs * 5 + params.tension_window * 5) + 5)

    def indicators(self, data, params):
        close = data["close"].astype(float)
        ret = close.pct_change()
        next_ret = ret.shift(-1)
        weekday = pd.Series(data.index.dayofweek, index=data.index)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        regime_ok = (close > ma).astype(float)

        seasonal_mean = pd.Series(np.nan, index=data.index, dtype=float)
        count_by_day = pd.Series(0.0, index=data.index, dtype=float)

        for d in range(5):
            mask = (weekday == d)
            if not mask.any():
                continue
            nr_d = next_ret.where(mask)
            cs = nr_d.fillna(0.0).cumsum()
            cnt = mask.astype(int).cumsum()
            past_cs = cs - nr_d.fillna(0.0)
            past_cnt = cnt - mask.astype(int)
            mean_past = past_cs / past_cnt.where(past_cnt > 0)
            seasonal_mean = seasonal_mean.where(~mask, mean_past)
            count_by_day = count_by_day.where(~mask, past_cnt.astype(float))

        dev = next_ret - seasonal_mean

        tension = pd.Series(np.nan, index=data.index, dtype=float)
        for d in range(5):
            mask = (weekday == d)
            if not mask.any():
                continue
            subset = dev[mask]
            if subset.empty:
                continue
            roll = subset.rolling(params.tension_window, min_periods=params.tension_window).sum()
            roll_past = roll.shift(1)
            full = roll_past.reindex(data.index)
            tension = tension.where(~mask, full)

        return pd.DataFrame(
            {
                "regime_ok": regime_ok,
                "seasonal_mean": seasonal_mean,
                "tension": tension,
                "weekday_obs_count": count_by_day,
            },
            index=data.index,
        )

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        regime_ok = indicators["regime_ok"].to_numpy(dtype=float)
        seasonal_mean = indicators["seasonal_mean"].to_numpy(dtype=float)
        tension = indicators["tension"].to_numpy(dtype=float)
        obs_count = indicators["weekday_obs_count"].to_numpy(dtype=float)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                if gain >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    raw[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                else:
                    raw[i] = 1
            else:
                tens_ok = (not np.isnan(tension[i])) and tension[i] <= params.tension_threshold
                sm_ok = (not np.isnan(seasonal_mean[i])) and seasonal_mean[i] > 0.0
                regime = (not np.isnan(regime_ok[i])) and regime_ok[i] >= 0.5
                obs_ok = (not np.isnan(obs_count[i])) and obs_count[i] >= params.seasonal_min_obs
                if tens_ok and sm_ok and regime and obs_ok:
                    raw[i] = 1
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = float(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
