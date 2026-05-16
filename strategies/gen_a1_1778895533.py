from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringReleaseParams:
    ma_len: int = 20
    std_len: int = 20
    z_pctl_len: int = 100
    low_pctl: float = 15.0
    high_pctl: float = 85.0
    dd_len: int = 60
    dd_min: float = 0.03
    atr_len: int = 14
    breakeven_pct: float = 0.02
    trail_k: float = 2.5
    init_stop_k: float = 3.0
    max_hold: int = 5
    size_scale: float = 1.0


class GeneratedStrategy(BaseStrategy[SpringReleaseParams]):
    strategy_id = "gen_a1_1778895533"

    @classmethod
    def params_type(cls):
        return SpringReleaseParams

    @staticmethod
    def warmup_bars(params: SpringReleaseParams) -> int:
        base = params.z_pctl_len + max(params.ma_len, params.std_len) + 2
        return int(max(base, params.dd_len + 1, params.atr_len + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SpringReleaseParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_len).mean()
        std = close.rolling(params.std_len).std()
        std_safe = std.where(std > 0)
        z = (close - ma) / std_safe

        z_vel = z.diff()
        z_low = z.rolling(params.z_pctl_len).quantile(params.low_pctl / 100.0)
        z_high = z.rolling(params.z_pctl_len).quantile(params.high_pctl / 100.0)

        roll_max = close.rolling(params.dd_len).max()
        drawdown = roll_max / close - 1.0

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        ind = pd.DataFrame(index=data.index)
        ind["z"] = z
        ind["z_vel"] = z_vel
        ind["z_low"] = z_low
        ind["z_high"] = z_high
        ind["drawdown"] = drawdown
        ind["atr"] = atr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringReleaseParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        z = indicators["z"].to_numpy(dtype=float)
        z_vel = indicators["z_vel"].to_numpy(dtype=float)
        z_low = indicators["z_low"].to_numpy(dtype=float)
        z_high = indicators["z_high"].to_numpy(dtype=float)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        state = 0
        in_pos = False
        direction = 0
        entry_price = 0.0
        stop = 0.0
        be_hit = False
        bars_held = 0

        dd_short = params.dd_min * 0.5

        for i in range(n):
            c = close[i]
            h = high[i]
            l = low[i]
            zi = z[i]
            zv = z_vel[i]
            zlo = z_low[i]
            zhi = z_high[i]
            dd = drawdown[i]
            a = atr[i]

            valid = not (
                np.isnan(zi)
                or np.isnan(zv)
                or np.isnan(zlo)
                or np.isnan(zhi)
                or np.isnan(dd)
                or np.isnan(a)
                or a <= 0.0
            )

            if in_pos:
                bars_held += 1
                exit_now = False
                if direction == 1:
                    if l <= stop:
                        exit_now = True
                else:
                    if h >= stop:
                        exit_now = True
                if bars_held >= params.max_hold:
                    exit_now = True

                if exit_now:
                    raw[i] = 0
                    in_pos = False
                    direction = 0
                    be_hit = False
                    bars_held = 0
                    state = 0
                    continue

                raw[i] = direction
                if valid:
                    if direction == 1:
                        if (not be_hit) and h >= entry_price * (1.0 + params.breakeven_pct):
                            be_hit = True
                            if entry_price > stop:
                                stop = entry_price
                        if be_hit:
                            cand = c - params.trail_k * a
                            if cand > stop:
                                stop = cand
                    else:
                        if (not be_hit) and l <= entry_price * (1.0 - params.breakeven_pct):
                            be_hit = True
                            if entry_price < stop:
                                stop = entry_price
                        if be_hit:
                            cand = c + params.trail_k * a
                            if cand < stop:
                                stop = cand
                continue

            if not valid:
                raw[i] = 0
                continue

            if state == 0:
                if zi <= zlo and zi < 0.0 and dd >= params.dd_min:
                    state = 1
                elif zi >= zhi and zi > 0.0 and dd <= dd_short:
                    state = -1
            elif state == 1:
                if zi > 0.0:
                    state = 0
                elif zv > 0.0:
                    direction = 1
                    in_pos = True
                    entry_price = c
                    be_hit = False
                    bars_held = 0
                    stop = c - params.init_stop_k * a
                    disp = min(abs(zi), 4.0)
                    sz = 0.6 + params.size_scale * (disp / 4.0)
                    size[i] = float(min(max(sz, 0.5), 1.5))
                    raw[i] = 1
                    state = 0
                    continue
            elif state == -1:
                if zi < 0.0:
                    state = 0
                elif zv < 0.0:
                    direction = -1
                    in_pos = True
                    entry_price = c
                    be_hit = False
                    bars_held = 0
                    stop = c + params.init_stop_k * a
                    disp = min(abs(zi), 4.0)
                    sz = 0.6 + params.size_scale * (disp / 4.0)
                    size[i] = float(min(max(sz, 0.5), 1.5))
                    raw[i] = -1
                    state = 0
                    continue

            raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
