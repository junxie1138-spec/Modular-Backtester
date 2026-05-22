from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilAccumulationParams:
    vol_window: int = 20
    rank_window: int = 252
    ma_window: int = 200
    enter_pct: float = 0.20
    exit_pct: float = 0.55


class GeneratedStrategy(BaseStrategy[CoilAccumulationParams]):
    strategy_id = "gen_a1_1778908300"

    @classmethod
    def params_type(cls) -> type[CoilAccumulationParams]:
        return CoilAccumulationParams

    def warmup_bars(self, params: CoilAccumulationParams) -> int:
        # comp_rank needs rank_window observations of norm_range,
        # and norm_range itself needs vol_window bars of ATR; +1 for the
        # prev-close shift inside true range.
        return int(max(params.rank_window + params.vol_window, params.ma_window) + 1)

    def indicators(self, data: pd.DataFrame, params: CoilAccumulationParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(
            params.vol_window, min_periods=params.vol_window
        ).mean()
        # Scale-free range measure so the percentile rank is comparable
        # across price levels over a multi-year lookback.
        norm_range = atr / close.replace(0.0, np.nan)
        comp_rank = norm_range.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        sma = close.rolling(
            params.ma_window, min_periods=params.ma_window
        ).mean()
        regime_ok = (close > sma) & sma.notna()

        out = pd.DataFrame(index=data.index)
        out["true_range"] = true_range
        out["atr"] = atr
        out["norm_range"] = norm_range
        out["comp_rank"] = comp_rank
        out["sma"] = sma
        out["regime_ok"] = regime_ok
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilAccumulationParams,
    ) -> SignalFrame:
        comp = indicators["comp_rank"].to_numpy(dtype=float)
        regime = indicators["regime_ok"].to_numpy(dtype=bool)
        n = len(comp)
        raw = np.zeros(n, dtype=np.int64)

        enter_pct = float(params.enter_pct)
        exit_pct = float(params.exit_pct)
        # Hysteresis state machine: flat -> long when compression rank drops
        # into the bottom band inside a bull regime; long -> flat only when
        # the entry condition flips (rank re-expands past the upper band,
        # or the 200-day regime filter turns off). Signal-reversal exit.
        state = 0
        for i in range(n):
            c = comp[i]
            valid = c == c  # False when NaN (warmup)
            if state == 0:
                if regime[i] and valid and c <= enter_pct:
                    state = 1
            else:
                if (not regime[i]) or (valid and c >= exit_pct):
                    state = 0
            raw[i] = state

        df = pd.DataFrame(index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = (
            pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
