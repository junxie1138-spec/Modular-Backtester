from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    range_lookback: int = 60
    shock_percentile: float = 90.0
    close_position_min: float = 0.60
    vol_lookback: int = 20
    target_vol: float = 0.15
    atr_period: int = 14
    breakeven_pct: float = 0.03
    trail_atr_mult: float = 3.0
    initial_atr_mult: float = 2.0
    max_hold: int = 12
    size_floor: float = 0.25
    size_cap: float = 1.0


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_a1_1778885107"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return int(max(params.range_lookback, params.vol_lookback, params.atr_period)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        rng = high - low
        safe_rng = rng.where(rng > 0.0, np.nan)
        close_pos = ((close - low) / safe_rng).fillna(0.5).clip(0.0, 1.0)

        q = float(np.clip(params.shock_percentile, 1.0, 99.0)) / 100.0
        shock_threshold = tr.rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).quantile(q)

        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        rets = close.pct_change()
        realized_vol = rets.rolling(
            params.vol_lookback, min_periods=params.vol_lookback
        ).std() * np.sqrt(252.0)

        vt_size = params.target_vol / realized_vol.replace(0.0, np.nan)
        vt_size = vt_size.fillna(params.size_floor).clip(params.size_floor, params.size_cap)

        shock = (tr >= shock_threshold) & (close_pos >= params.close_position_min)
        shock = shock.fillna(False)

        out = pd.DataFrame(index=data.index)
        out["tr"] = tr
        out["atr"] = atr
        out["close_pos"] = close_pos
        out["shock"] = shock.astype(bool)
        out["vt_size"] = vt_size
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        shock = indicators["shock"].to_numpy(dtype=bool)
        n = len(close)

        raw = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        bars_held = 0
        be_armed = False

        for i in range(n):
            if not in_pos:
                if (
                    shock[i]
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                    and np.isfinite(close[i])
                ):
                    in_pos = True
                    entry_price = close[i]
                    peak = close[i]
                    stop = entry_price - params.initial_atr_mult * atr[i]
                    bars_held = 0
                    be_armed = False
                    raw[i] = 1
            else:
                bars_held += 1
                px = close[i]
                if np.isfinite(px) and px > peak:
                    peak = px

                a = atr[i] if np.isfinite(atr[i]) else 0.0

                if (not be_armed) and peak >= entry_price * (1.0 + params.breakeven_pct):
                    be_armed = True

                trail_stop = peak - params.trail_atr_mult * a
                cand = max(trail_stop, entry_price) if be_armed else trail_stop
                if cand > stop:
                    stop = cand

                exit_now = (px <= stop) or (bars_held >= params.max_hold)
                if exit_now:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["vt_size"].astype(float)
        size = size.fillna(params.size_floor).clip(params.size_floor, params.size_cap)
        df["size"] = size.to_numpy(dtype=float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
