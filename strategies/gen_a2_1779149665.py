from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEpidemicParams:
    gap_win: int = 60
    regime_win: int = 40
    regime_thr: float = 0.5
    gap_floor: float = 0.0015
    gap_z_lo: float = 0.4
    gap_z_hi: float = 3.0
    atr_len: int = 14
    stop_k: float = 2.5
    max_hold: int = 10
    ma_len: int = 200
    base_size: float = 1.0
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[GapEpidemicParams]):
    strategy_id = "gen_a2_1779149665"

    @classmethod
    def params_type(cls):
        return GapEpidemicParams

    def warmup_bars(self, params: GapEpidemicParams) -> int:
        return int(max(params.ma_len, params.gap_win,
                       params.regime_win, params.atr_len)) + 2

    def indicators(self, data: pd.DataFrame,
                   params: GapEpidemicParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        prior_close = close.shift(1)
        gap = (open_ - prior_close) / prior_close

        gap_mean = gap.rolling(params.gap_win).mean()
        gap_std = gap.rolling(params.gap_win).std()
        gap_z = (gap - gap_mean) / gap_std.where(gap_std > 0)

        # Epidemic SI regime classifier: among recent up-gap "exposures",
        # the fraction whose gap stayed "infected" (close held above the open).
        up_gap = (gap > params.gap_floor).astype(float)
        held = ((gap > params.gap_floor) & (close > open_)).astype(float)
        exposed_sum = up_gap.rolling(params.regime_win).sum()
        held_sum = held.rolling(params.regime_win).sum()
        infection_ratio = (held_sum / exposed_sum.where(exposed_sum > 0)).fillna(0.5)

        tr = pd.concat([
            (high - low),
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        ma = close.rolling(params.ma_len).mean()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_z"] = gap_z
        out["infection_ratio"] = infection_ratio
        out["atr"] = atr
        out["ma"] = ma
        out["prior_close"] = prior_close
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: GapEpidemicParams) -> SignalFrame:
        p = params
        close = data["close"].to_numpy(dtype=float)
        open_ = data["open"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        gap_z = indicators["gap_z"].to_numpy(dtype=float)
        infection = indicators["infection_ratio"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        prior_close = indicators["prior_close"].to_numpy(dtype=float)
        n = len(close)

        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        span = max(p.gap_z_hi - p.gap_z_lo, 1e-9)

        position = 0
        entry_stop = 0.0
        bars_held = 0
        cur_size = p.base_size

        for i in range(n):
            if position == 1:
                bars_held += 1
                if close[i] <= entry_stop or bars_held >= p.max_hold:
                    position = 0
                    sig[i] = 0
                    size[i] = cur_size
                else:
                    sig[i] = 1
                    size[i] = cur_size
                continue

            valid = (np.isfinite(gap[i]) and np.isfinite(gap_z[i])
                     and np.isfinite(infection[i]) and np.isfinite(atr[i])
                     and np.isfinite(ma[i]) and np.isfinite(prior_close[i])
                     and atr[i] > 0.0)
            if not valid:
                continue

            momentum = infection[i] > p.regime_thr
            entered = False
            if momentum:
                # Infected regime: ride an up-gap that holds above its open.
                if (gap[i] > p.gap_floor
                        and p.gap_z_lo <= gap_z[i] <= p.gap_z_hi
                        and close[i] > open_[i] and close[i] > ma[i]):
                    entered = True
            else:
                # Susceptible regime: buy a down-gap the market absorbs back
                # above the prior close (the gap "heals").
                if (gap[i] < -p.gap_floor
                        and p.gap_z_lo <= -gap_z[i] <= p.gap_z_hi
                        and close[i] > prior_close[i] and close[i] > ma[i]):
                    entered = True

            if entered:
                strength = (abs(gap_z[i]) - p.gap_z_lo) / span
                strength = min(1.0, max(0.0, strength))
                cur_size = p.min_size + (p.base_size - p.min_size) * strength
                position = 1
                bars_held = 0
                entry_stop = close[i] - p.stop_k * atr[i]
                sig[i] = 1
                size[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.base_size)
        df["size"] = df["size"].clip(lower=p.min_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
