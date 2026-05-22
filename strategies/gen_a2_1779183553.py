from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_len: int = 20
    z_win: int = 20
    eff_win: int = 10
    eff_thresh: float = 0.5
    z_min: float = 0.5
    atr_len: int = 14
    k_atr: float = 2.5
    be_trigger: float = 0.015
    regime_len: int = 200
    vol_win: int = 20
    vol_target: float = 0.15


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779183553"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.regime_len + params.z_win + params.eff_win + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        ind = pd.DataFrame(index=data.index)

        # distance-from-MA z-score: how far close sits from its moving average,
        # standardised by the rolling dispersion of that distance.
        sma = close.rolling(params.ma_len).mean()
        dist = close - sma
        dist_std = dist.rolling(params.z_win).std()
        z = dist / dist_std.replace(0.0, np.nan)
        ind["z"] = z

        # signal-to-noise of the z-score's OWN path: directional efficiency =
        # net change over the window divided by total variation along the way.
        net = z - z.shift(params.eff_win)
        tv = z.diff().abs().rolling(params.eff_win).sum()
        eff = net / tv.replace(0.0, np.nan)
        ind["eff"] = eff

        # ATR for the breakeven-then-trail exit.
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
        ind["atr"] = atr

        # hard twist: 200-day MA regime filter.
        sma200 = close.rolling(params.regime_len).mean()
        regime = (close > sma200).astype(float)
        ind["regime"] = regime

        entry = (
            (eff > params.eff_thresh)
            & (z > params.z_min)
            & (net > 0.0)
            & (regime > 0.5)
        )
        ind["entry_cond"] = entry.fillna(False).astype(float)

        # volatility-targeted position size.
        rv = close.pct_change().rolling(params.vol_win).std() * np.sqrt(252.0)
        size = (params.vol_target / rv.replace(0.0, np.nan)).clip(0.3, 1.0)
        ind["size"] = size.fillna(0.5)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)
        entry_cond = indicators["entry_cond"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        desired = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False

        for i in range(n):
            c = close[i]
            a = atr[i]
            if position == 0:
                if entry_cond[i] >= 0.5 and np.isfinite(a) and a > 0.0:
                    position = 1
                    entry_price = c
                    stop = c - params.k_atr * a
                    armed = False
                    desired[i] = 1
            else:
                # breakeven: once price clears +be_trigger, lift stop to entry.
                if not armed and c >= entry_price * (1.0 + params.be_trigger):
                    stop = max(stop, entry_price)
                    armed = True
                # trail: after breakeven, ratchet the stop up by k*ATR.
                if armed and np.isfinite(a) and a > 0.0:
                    stop = max(stop, c - params.k_atr * a)
                if c <= stop:
                    position = 0
                    desired[i] = 0
                else:
                    desired[i] = 1

        out = pd.DataFrame(index=data.index)
        sig = pd.Series(desired, index=data.index)
        out["signal"] = sig.shift(1).fillna(0).astype(int)
        size = indicators["size"].reindex(data.index).fillna(0.5).clip(0.3, 1.0)
        out["size"] = size.shift(1).fillna(0.5).astype(float)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
