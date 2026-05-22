from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeUniformityParams:
    cv_window: int = 10
    pctl_lookback: int = 100
    entry_pctl: float = 0.20
    clv_min: float = 0.55
    hold_bars: int = 4
    trend_ma: int = 200
    vol_window: int = 20
    target_vol: float = 0.012
    use_trend_gate: bool = True


class GeneratedStrategy(BaseStrategy[RangeUniformityParams]):
    strategy_id = "gen_a2_1779150233"

    @classmethod
    def params_type(cls) -> type[RangeUniformityParams]:
        return RangeUniformityParams

    def warmup_bars(self, params: RangeUniformityParams) -> int:
        return int(
            max(
                params.trend_ma,
                params.pctl_lookback + params.cv_window,
                params.vol_window,
            )
        ) + 2

    def indicators(
        self, data: pd.DataFrame, params: RangeUniformityParams
    ) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = (high - low).astype(float)
        rmean = rng.rolling(params.cv_window).mean()
        rstd = rng.rolling(params.cv_window).std()
        cv = rstd / rmean.where(rmean > 0)
        cv_pctl = cv.rolling(params.pctl_lookback).rank(pct=True)

        rng_safe = rng.where(rng > 0)
        clv = ((close - low) / rng_safe).fillna(0.5)

        ma = close.rolling(params.trend_ma).mean()
        rvol = close.pct_change().rolling(params.vol_window).std()

        out = pd.DataFrame(index=data.index)
        out["range"] = rng
        out["cv"] = cv
        out["cv_pctl"] = cv_pctl
        out["clv"] = clv
        out["ma"] = ma
        out["rvol"] = rvol
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeUniformityParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].astype(float)

        cv_pctl = indicators["cv_pctl"]
        clv = indicators["clv"]
        ma = indicators["ma"]

        entry_cond = (cv_pctl <= params.entry_pctl) & (clv >= params.clv_min)
        if params.use_trend_gate:
            entry_cond = entry_cond & (close > ma)
        entry = entry_cond.fillna(False).to_numpy()

        hold = max(1, int(params.hold_bars))
        raw = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if entry[i]:
                end = min(i + hold, n)
                raw[i:end] = 1
                i = end
            else:
                i += 1

        rvol = indicators["rvol"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            size_raw = params.target_vol / rvol
        size_raw = np.where(np.isfinite(size_raw), size_raw, 1.0)
        size = np.clip(size_raw, 0.5, 1.5)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
