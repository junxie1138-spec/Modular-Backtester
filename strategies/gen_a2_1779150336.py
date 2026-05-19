from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_lookback: int = 60
    atr_period: int = 14
    vov_window: int = 20
    pct_window: int = 120
    vov_hi_pct: float = 0.80
    dd_pct: float = 0.70
    hold_bars: int = 2
    vol_lookback: int = 20
    target_vol: float = 0.16


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779150336"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        chain = params.atr_period + params.vov_window + params.pct_window
        return int(max(params.dd_lookback, chain, params.vol_lookback)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        # Volatility-of-volatility: dispersion of ATR itself.
        vov = atr.rolling(params.vov_window).std()
        vov_rank = vov.rolling(params.pct_window).rank(pct=True)

        # Percentile-ranked drawdown depth from a rolling peak.
        roll_max = close.rolling(params.dd_lookback).max()
        dd_depth = (roll_max - close) / roll_max.replace(0.0, np.nan)
        dd_rank = dd_depth.rolling(params.pct_window).rank(pct=True)

        # Percentile-ranked drawup from a rolling trough.
        roll_min = close.rolling(params.dd_lookback).min()
        du_depth = (close - roll_min) / roll_min.replace(0.0, np.nan)
        du_rank = du_depth.rolling(params.pct_window).rank(pct=True)

        ret = close.pct_change()
        ann_vol = ret.rolling(params.vol_lookback).std() * np.sqrt(252.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["vov"] = vov
        out["vov_rank"] = vov_rank
        out["dd_rank"] = dd_rank
        out["du_rank"] = du_rank
        out["ann_vol"] = ann_vol
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        df = pd.DataFrame(index=data.index)

        vov_rank = indicators["vov_rank"]
        prev_vov_rank = vov_rank.shift(1)

        # Roll-over: vol-of-vol leaves its top percentile band.
        cross_down = (prev_vov_rank >= params.vov_hi_pct) & (
            vov_rank < params.vov_hi_pct
        )

        deep_dd = indicators["dd_rank"] >= params.dd_pct
        deep_du = indicators["du_rank"] >= params.dd_pct

        long_entry = (cross_down & deep_dd).fillna(False).to_numpy()
        short_entry = (cross_down & deep_du).fillna(False).to_numpy()

        hold = max(1, int(params.hold_bars))
        raw = np.zeros(n, dtype=float)
        direction = 0
        bars_left = 0
        for i in range(n):
            if bars_left > 0:
                raw[i] = direction
                bars_left -= 1
                continue
            if long_entry[i]:
                direction = 1
                raw[i] = 1.0
                bars_left = hold - 1
            elif short_entry[i]:
                direction = -1
                raw[i] = -1.0
                bars_left = hold - 1
            else:
                direction = 0

        sig = pd.Series(raw, index=data.index)
        df["signal"] = sig.shift(1).fillna(0).astype(int)

        ann_vol = indicators["ann_vol"].replace(0.0, np.nan)
        size = (params.target_vol / ann_vol).clip(lower=0.3, upper=1.5)
        df["size"] = size.fillna(1.0).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
