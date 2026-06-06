from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyStreakParams:
    sma_window: int = 150
    atr_window: int = 14
    streak_lookback: int = 8
    entry_threshold: float = 0.05
    stop_atr_mult: float = 2.0
    max_hold: int = 2
    base_size: float = 1.0
    size_gain: float = 1.5
    min_size: float = 0.4
    max_size: float = 1.6
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[PredatorPreyStreakParams]):
    strategy_id = "gen_a1_1778905079"

    @classmethod
    def params_type(cls):
        return PredatorPreyStreakParams

    @staticmethod
    def warmup_bars(params: PredatorPreyStreakParams) -> int:
        return int(
            params.sma_window
            + params.atr_window
            + 4 * max(int(params.streak_lookback), 1)
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: PredatorPreyStreakParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        # --- consecutive streak counts -------------------------------------
        up = close > prev_close
        down = close < prev_close
        up_int = up.astype(int)
        down_int = down.astype(int)
        up_streak = up_int.groupby((up_int == 0).cumsum()).cumsum()
        down_streak = down_int.groupby((down_int == 0).cumsum()).cumsum()

        # --- completed-streak lengths (predator-prey populations) ----------
        prev_down_streak = down_streak.shift(1)
        prev_up_streak = up_streak.shift(1)
        # a down-streak completes on the bar where price closes up again
        dc_len = prev_down_streak.where(up & (prev_down_streak > 0))
        # an up-streak completes on the bar where price closes down again
        uc_len = prev_up_streak.where(down & (prev_up_streak > 0))

        k = max(int(params.streak_lookback), 2)
        dc_obs = dc_len.dropna()
        uc_obs = uc_len.dropna()
        mean_down = (
            dc_obs.rolling(k, min_periods=2).mean().reindex(data.index).ffill()
        )
        mean_up = (
            uc_obs.rolling(k, min_periods=2).mean().reindex(data.index).ffill()
        )

        denom = (mean_up + mean_down).replace(0.0, np.nan)
        # positive => prey-dominant (up-streaks historically longer)
        regime_strength = (mean_up - mean_down) / denom

        # --- ATR ------------------------------------------------------------
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(max(int(params.atr_window), 1)).mean()

        sma = close.rolling(max(int(params.sma_window), 1)).mean()

        out = pd.DataFrame(index=data.index)
        out["up_streak"] = up_streak
        out["regime_strength"] = regime_strength
        out["atr"] = atr
        out["sma"] = sma
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"]
        up_streak = indicators["up_streak"]
        regime_strength = indicators["regime_strength"].fillna(-1.0)
        atr = indicators["atr"]
        sma = indicators["sma"]

        if params.use_trend_filter:
            trend_ok = (close > sma).fillna(False)
        else:
            trend_ok = pd.Series(True, index=data.index)

        # entry: newborn up-streak in a prey-dominant regime
        entry_cond = (
            (up_streak == 1)
            & (regime_strength > params.entry_threshold)
            & trend_ok
        )

        # signal-scaled size: stronger prey dominance -> larger position
        size_series = (
            params.base_size * (1.0 + params.size_gain * regime_strength)
        ).clip(lower=params.min_size, upper=params.max_size)

        close_arr = close.to_numpy(dtype=float)
        atr_arr = atr.to_numpy(dtype=float)
        entry_arr = entry_cond.to_numpy(dtype=bool)
        size_arr = size_series.to_numpy(dtype=float)

        n = len(close_arr)
        sig = np.zeros(n, dtype=int)
        base_sz = max(float(params.base_size), 1e-6)
        sz = np.full(n, base_sz, dtype=float)

        in_pos = False
        stop = -np.inf
        held = 0
        cur_size = base_sz

        for i in range(n):
            if in_pos:
                held += 1
                # fixed volatility-stop (set at entry) OR holding-horizon stop
                exit_now = (close_arr[i] < stop) or (held >= int(params.max_hold))
                if exit_now:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1
                    sz[i] = cur_size
            if (not in_pos) and entry_arr[i]:
                in_pos = True
                entry_price = close_arr[i]
                a = atr_arr[i]
                if np.isnan(a) or a <= 0.0:
                    stop = -np.inf
                else:
                    stop = entry_price - params.stop_atr_mult * a
                held = 0
                s = size_arr[i]
                cur_size = float(s) if np.isfinite(s) and s > 0.0 else base_sz
                sig[i] = 1
                sz[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = sz
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(base_sz)
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
