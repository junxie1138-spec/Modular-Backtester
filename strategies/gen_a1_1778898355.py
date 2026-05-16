from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolRankParams:
    vol_window: int = 20
    rank_window: int = 252
    low_pct: float = 0.30
    trend_window: int = 100
    atr_window: int = 14
    atr_stop_mult: float = 2.5
    max_hold: int = 10
    target_vol: float = 0.15
    size_min: float = 0.25
    size_max: float = 1.5


class GeneratedStrategy(BaseStrategy[VolRankParams]):
    strategy_id = "gen_a1_1778898355"

    @classmethod
    def params_type(cls) -> type[VolRankParams]:
        return VolRankParams

    @staticmethod
    def warmup_bars(params: VolRankParams) -> int:
        return int(params.vol_window + params.rank_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolRankParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        rvol = ret.rolling(params.vol_window).std()
        vol_rank = rvol.rolling(params.rank_window).rank(pct=True)
        sma = close.rolling(params.trend_window).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ann_vol = rvol * np.sqrt(252.0)
        raw_entry = ((vol_rank < params.low_pct) & (close > sma)).astype(float)

        out = pd.DataFrame(index=data.index)
        out["rvol"] = rvol
        out["vol_rank"] = vol_rank
        out["atr"] = atr
        out["ann_vol"] = ann_vol
        out["raw_entry"] = raw_entry
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolRankParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = indicators["raw_entry"].fillna(0.0).to_numpy(dtype=float)
        n = len(data)

        # Two-bar confirmation: compression must hold on this bar and the prior bar.
        confirmed = np.zeros(n, dtype=bool)
        for i in range(1, n):
            confirmed[i] = (raw[i] > 0.5) and (raw[i - 1] > 0.5)

        signal = np.zeros(n, dtype=int)
        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if confirmed[i] and not np.isnan(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    stop_level = close[i] - params.atr_stop_mult * atr[i]
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                # Fixed volatility-stop: stop_level frozen at entry, not trailed.
                if close[i] <= stop_level or bars_held >= params.max_hold:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal

        # Volatility-targeting: size inversely scaled to annualized realized vol.
        ann_vol = indicators["ann_vol"].to_numpy(dtype=float)
        valid = (ann_vol > 1e-6) & ~np.isnan(ann_vol)
        size = np.divide(
            params.target_vol,
            np.where(valid, ann_vol, np.nan),
            out=np.full(n, np.nan),
            where=valid,
        )
        size = np.clip(size, params.size_min, params.size_max)
        size = np.where(np.isnan(size), 1.0, size)
        df["size"] = size

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
