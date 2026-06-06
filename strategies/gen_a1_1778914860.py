from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSurvivalParams:
    gap_min_pct: float = 0.0
    vol_pct_window: int = 60
    vol_pct_threshold: float = 0.80
    atr_window: int = 14
    atr_k: float = 2.5
    max_hold: int = 5
    trend_ma: int = 200
    use_trend_gate: bool = True


class GeneratedStrategy(BaseStrategy[GapSurvivalParams]):
    strategy_id = "gen_a1_1778914860"

    @classmethod
    def params_type(cls):
        return GapSurvivalParams

    def warmup_bars(self, params: GapSurvivalParams) -> int:
        return int(max(params.vol_pct_window, params.atr_window + 1, params.trend_ma)) + 1

    def indicators(self, data: pd.DataFrame, params: GapSurvivalParams) -> pd.DataFrame:
        o = data["open"]
        h = data["high"]
        l = data["low"]
        c = data["close"]
        v = data["volume"]

        prev_close = c.shift(1)
        gap = (o - prev_close) / prev_close

        # Up-gap that was never filled intraday (low stayed above prior close).
        gap_up = gap > params.gap_min_pct
        gap_survived = l >= prev_close

        # Wilder-style true range -> simple ATR.
        tr = pd.concat(
            [
                h - l,
                (h - prev_close).abs(),
                (l - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        # Volume percentile rank within trailing window (capacity-overflow trigger).
        vol_rank = v.rolling(params.vol_pct_window).rank(pct=True)
        vol_ok = vol_rank >= params.vol_pct_threshold

        ma = c.rolling(params.trend_ma).mean()
        if params.use_trend_gate:
            trend_ok = c > ma
        else:
            trend_ok = pd.Series(True, index=data.index)

        entry_ok = (gap_up & gap_survived & vol_ok & trend_ok).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["prev_close"] = prev_close
        out["gap"] = gap
        out["atr"] = atr
        out["vol_rank"] = vol_rank
        out["ma"] = ma
        out["entry_ok"] = entry_ok.astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSurvivalParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_ok = indicators["entry_ok"].to_numpy()
        n = len(close)

        pos = np.zeros(n, dtype=int)
        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if (
                    bool(entry_ok[i])
                    and np.isfinite(atr[i])
                    and np.isfinite(close[i])
                ):
                    in_pos = True
                    # Fixed (non-trailing) volatility stop set once at entry.
                    stop_level = close[i] - params.atr_k * atr[i]
                    bars_held = 0
                    pos[i] = 1
            else:
                bars_held += 1
                if close[i] < stop_level or bars_held >= params.max_hold:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
