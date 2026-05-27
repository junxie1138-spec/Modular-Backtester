from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_lookback: int = 15
    entry_threshold: float = 0.40
    profit_target_pct: float = 0.06
    max_hold_bars: int = 20
    trend_ma: int = 200
    min_size: float = 0.30
    max_size: float = 0.95
    size_scale_cap: float = 1.00


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779879182"

    @classmethod
    def params_type(cls) -> type:
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(int(params.trend_ma), int(params.gap_lookback)) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        safe_prev_close = prev_close.replace(0, np.nan)
        gap = (open_ - prev_close) / safe_prev_close
        gap = gap.replace([np.inf, -np.inf], np.nan)
        positive_gap = gap.clip(lower=0.0).fillna(0.0)

        safe_close = close.replace(0, np.nan)
        range_norm = (high - low) / safe_close
        range_norm = range_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # zero-out the first bar where prev_close is undefined to keep window aligned
        range_norm.iloc[0] = 0.0

        L = int(params.gap_lookback)
        pos_gap_sum = positive_gap.rolling(L, min_periods=L).sum()
        range_sum = range_norm.rolling(L, min_periods=L).sum()

        utilization = pos_gap_sum / range_sum.replace(0, np.nan)
        utilization = utilization.replace([np.inf, -np.inf], np.nan)

        ma_w = int(params.trend_ma)
        trend_ma = close.rolling(ma_w, min_periods=ma_w).mean()
        trend_ok = (close > trend_ma).astype(float)
        trend_ok = trend_ok.where(trend_ma.notna(), np.nan)

        out = pd.DataFrame(
            {
                "utilization": utilization,
                "trend_ok": trend_ok,
                "gap": gap,
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
        max_size = float(params.max_size)
        min_size = float(params.min_size)
        signal = np.zeros(n, dtype=np.int64)
        size = np.full(n, max_size, dtype=np.float64)

        utilization = indicators["utilization"].to_numpy(dtype=np.float64, copy=False)
        trend_ok = indicators["trend_ok"].to_numpy(dtype=np.float64, copy=False)
        close = data["close"].to_numpy(dtype=np.float64, copy=False)

        threshold = float(params.entry_threshold)
        cap = float(params.size_scale_cap)
        denom = cap - threshold
        if denom <= 0.0:
            denom = 1e-9
        size_span = max_size - min_size
        if size_span < 0.0:
            size_span = 0.0
        target = float(params.profit_target_pct)
        max_hold = int(params.max_hold_bars)

        in_trade = False
        entry_price = 0.0
        bars_held = 0
        entry_size = max_size

        for i in range(n):
            if in_trade:
                bars_held += 1
                pnl = 0.0
                if entry_price > 0.0:
                    pnl = (close[i] - entry_price) / entry_price
                if pnl >= target or bars_held >= max_hold:
                    in_trade = False
                    entry_price = 0.0
                    bars_held = 0
                    entry_size = max_size
                    signal[i] = 0
                    size[i] = max_size
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                u = utilization[i]
                t = trend_ok[i]
                u_prev = utilization[i - 1] if i > 0 else np.nan
                crossed_up = (
                    np.isfinite(u)
                    and np.isfinite(t)
                    and t > 0.5
                    and u >= threshold
                    and ((not np.isfinite(u_prev)) or u_prev < threshold)
                )
                if crossed_up:
                    u_clipped = u
                    if u_clipped < threshold:
                        u_clipped = threshold
                    if u_clipped > cap:
                        u_clipped = cap
                    frac = (u_clipped - threshold) / denom
                    s = min_size + size_span * frac
                    if s <= 0.0:
                        s = min_size if min_size > 0.0 else max_size
                    entry_size = s
                    in_trade = True
                    entry_price = float(close[i])
                    bars_held = 0
                    signal[i] = 1
                    size[i] = s
                else:
                    signal[i] = 0
                    size[i] = max_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(max_size)
        df.loc[df["size"] <= 0, "size"] = max_size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
