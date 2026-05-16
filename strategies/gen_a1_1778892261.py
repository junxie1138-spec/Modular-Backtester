from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyParams:
    ma_len: int = 20
    z_len: int = 40
    acf_win: int = 60
    acf_lag: int = 8
    acf_thresh: float = 0.15
    entry_z: float = 1.5
    regime_ma: int = 200
    profit_target: float = 0.05
    time_stop: int = 10


class GeneratedStrategy(BaseStrategy[PredatorPreyParams]):
    """Predator-prey cycle-gated reversion on the distance-from-MA z-score.

    The z-score of price's distance from its moving average is treated as a
    'prey population' that booms above fair value and crashes below it. A
    genuine predator-prey limit cycle leaves a measurable fingerprint: the
    autocorrelation of the z-score at a half-cycle lag turns negative. Only
    when that fingerprint is present do we fade z-score extremes, and only
    in the direction sanctioned by the 200-day regime filter.
    """

    strategy_id = "gen_a1_1778892261"

    @classmethod
    def params_type(cls) -> type[PredatorPreyParams]:
        return PredatorPreyParams

    def warmup_bars(self, params: PredatorPreyParams) -> int:
        # z-score needs ma_len + z_len bars; its rolling ACF then needs a
        # further acf_win + acf_lag bars. Regime MA needs regime_ma bars.
        cyclic_need = params.ma_len + params.z_len + params.acf_win + params.acf_lag
        return int(max(params.regime_ma, cyclic_need)) + 1

    def indicators(self, data: pd.DataFrame, params: PredatorPreyParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)

        sma = close.rolling(params.ma_len).mean()
        dist = close - sma
        dist_mean = dist.rolling(params.z_len).mean()
        dist_std = dist.rolling(params.z_len).std()
        # distance-from-MA z-score (NaN-safe: zero std -> NaN, not inf)
        z = (dist - dist_mean) / dist_std.replace(0.0, np.nan)
        out["z"] = z

        # Lag-L autocorrelation of the z-score. A significantly negative
        # value is the signature of an oscillatory (predator-prey) regime:
        # the z-score sits on opposite sides of fair value ~acf_lag apart.
        out["acf"] = z.rolling(params.acf_win).corr(z.shift(params.acf_lag))

        out["sma200"] = close.rolling(params.regime_ma).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorPreyParams,
    ) -> SignalFrame:
        p = params
        df = pd.DataFrame(index=data.index)
        close = data["close"]

        z = indicators["z"]
        acf = indicators["acf"]
        sma200 = indicators["sma200"]

        # Cyclic (limit-cycle) regime: half-cycle ACF clearly negative.
        cyclic = acf < -p.acf_thresh
        # 200-day MA regime filter (the hard twist): only fade dips with
        # the long-term trend, only fade rallies against it.
        bull = close > sma200
        bear = close < sma200

        z_prev = z.shift(1)
        # Fresh crossings into z-score extremes (cycle trough / peak).
        long_cross = (z < -p.entry_z) & (z_prev >= -p.entry_z)
        short_cross = (z > p.entry_z) & (z_prev <= p.entry_z)

        long_entry = (cyclic & bull & long_cross).fillna(False).to_numpy()
        short_entry = (cyclic & bear & short_cross).fillna(False).to_numpy()

        px = close.to_numpy(dtype=float)
        n = len(df)
        sig = np.zeros(n, dtype=int)

        # Path-dependent exit: profit-target OR time-stop, whichever first.
        pos = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(n):
            if pos != 0:
                bars_held += 1
                if entry_price > 0.0:
                    pnl = (px[i] - entry_price) / entry_price * pos
                else:
                    pnl = 0.0
                if pnl >= p.profit_target or bars_held >= p.time_stop:
                    pos = 0
                    bars_held = 0
            if pos == 0:
                if long_entry[i]:
                    pos = 1
                    entry_price = px[i]
                    bars_held = 0
                elif short_entry[i]:
                    pos = -1
                    entry_price = px[i]
                    bars_held = 0
            sig[i] = pos

        df["signal"] = sig
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
