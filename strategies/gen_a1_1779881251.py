from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    autocorr_window: int = 30
    autocorr_rank_window: int = 252
    autocorr_rank_min: float = 0.70
    vol_window: int = 20
    vol_rank_window: int = 252
    vol_rank_max: float = 0.30
    hold_bars: int = 17
    spike_atr_window: int = 14
    spike_mult: float = 2.5
    refractory_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779881251"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        ac_need = params.autocorr_window + params.autocorr_rank_window + 2
        vol_need = params.vol_window + params.vol_rank_window + 2
        spike_need = params.spike_atr_window + 2
        return int(max(ac_need, vol_need, spike_need))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        returns = close.pct_change()

        r0 = returns
        r1 = returns.shift(1)
        autocorr = r0.rolling(params.autocorr_window, min_periods=params.autocorr_window).corr(r1)
        autocorr_rank = autocorr.rolling(
            params.autocorr_rank_window, min_periods=params.autocorr_rank_window
        ).rank(pct=True)

        rvol = returns.rolling(params.vol_window, min_periods=params.vol_window).std()
        vol_rank = rvol.rolling(
            params.vol_rank_window, min_periods=params.vol_rank_window
        ).rank(pct=True)

        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        atr = tr.rolling(params.spike_atr_window, min_periods=params.spike_atr_window).mean()
        spike_thresh = params.spike_mult * atr
        spike = (tr > spike_thresh).fillna(False).astype(np.int64)

        out = pd.DataFrame(
            {
                "autocorr": autocorr,
                "autocorr_rank": autocorr_rank,
                "rvol": rvol,
                "vol_rank": vol_rank,
                "atr": atr,
                "spike": spike,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=np.int64)

        ac_rank = indicators["autocorr_rank"].to_numpy(dtype=np.float64, copy=False)
        vol_rank = indicators["vol_rank"].to_numpy(dtype=np.float64, copy=False)
        spike = indicators["spike"].to_numpy(dtype=np.int64, copy=False)

        ac_min = float(params.autocorr_rank_min)
        vol_max = float(params.vol_rank_max)
        hold = int(params.hold_bars)
        refractory = int(params.refractory_bars)

        bars_left = 0
        refractory_left = 0

        for i in range(n):
            if refractory_left > 0:
                refractory_left -= 1
            if spike[i] == 1:
                refractory_left = refractory

            if bars_left > 0:
                signal[i] = 1
                bars_left -= 1
                continue

            ac_val = ac_rank[i]
            vol_val = vol_rank[i]
            if np.isnan(ac_val) or np.isnan(vol_val):
                continue
            if refractory_left > 0:
                continue

            if ac_val >= ac_min and vol_val <= vol_max:
                signal[i] = 1
                bars_left = hold - 1

        size = np.ones(n, dtype=np.float64)

        df = pd.DataFrame(
            {
                "signal": signal,
                "size": size,
            },
            index=data.index,
        )

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
