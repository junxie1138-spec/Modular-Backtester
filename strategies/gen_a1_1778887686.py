from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDrawdownParams:
    hwm_window: int = 60
    atr_window: int = 14
    pct_q: float = 0.80
    atr_k: float = 2.0
    max_hold: int = 2
    min_obs: int = 20


class GeneratedStrategy(BaseStrategy[SeasonalDrawdownParams]):
    """Weekday-conditioned drawdown-depth percentile entry, fixed ATR vol-stop."""

    strategy_id = "gen_a1_1778887686"

    @classmethod
    def params_type(cls):
        return SeasonalDrawdownParams

    def warmup_bars(self, params: SeasonalDrawdownParams) -> int:
        # high-water window, ATR (uses prev close), and enough same-weekday
        # observations (~5 trading days per weekday cycle) for the percentile.
        return int(max(params.hwm_window,
                       params.atr_window + 1,
                       params.min_obs * 5 + 5))

    def indicators(self, data: pd.DataFrame,
                   params: SeasonalDrawdownParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        # Drawdown depth from a rolling high-water mark (>= 0, NaN during warmup).
        hwm = close.rolling(params.hwm_window,
                            min_periods=params.hwm_window).max()
        depth = ((hwm - close) / hwm).clip(lower=0.0)

        # Average true range for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low,
             (high - prev_close).abs(),
             (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window,
                         min_periods=params.atr_window).mean()

        # Percentile threshold (the twist): for each weekday, an expanding
        # quantile of all PRIOR same-weekday drawdown depths. shift(1) inside
        # each group keeps the threshold strictly out-of-sample.
        dow = pd.Series(data.index.dayofweek, index=data.index)
        tmp = pd.DataFrame({"dow": dow, "depth": depth})

        q = float(params.pct_q)
        min_obs = int(params.min_obs)

        def _exp_q(s: pd.Series) -> pd.Series:
            return s.expanding(min_periods=min_obs).quantile(q).shift(1)

        thresh = tmp.groupby("dow")["depth"].transform(_exp_q)

        # Entry fires when today's depth ranks above this weekday's percentile.
        entry_raw = (
            (depth >= thresh)
            & (depth > 0.0)
            & thresh.notna()
            & depth.notna()
        ).astype(float)

        out = pd.DataFrame(index=data.index)
        out["depth"] = depth
        out["thresh"] = thresh
        out["atr"] = atr
        out["entry_raw"] = entry_raw.fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: SeasonalDrawdownParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        position = 0
        stop_level = 0.0
        bars_held = 0
        max_hold = int(params.max_hold)
        k = float(params.atr_k)

        for i in range(n):
            exited = False
            if position == 1:
                bars_held += 1
                # Fixed volatility stop set once at entry (not trailing),
                # or time-stop at the target holding horizon.
                if (close[i] <= stop_level) or (bars_held >= max_hold):
                    position = 0
                    exited = True
            if position == 0 and not exited:
                a = atr[i]
                if entry_raw[i] >= 1.0 and np.isfinite(a) and a > 0.0:
                    position = 1
                    entry_price = close[i]
                    stop_level = entry_price - k * a
                    bars_held = 0
            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = (pd.Series(signal, index=data.index)
                        .shift(1).fillna(0).astype(int))
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal",
                           size_column="size")
