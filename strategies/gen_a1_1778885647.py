from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) structural constants. Keeping these hard-coded satisfies
# the <=2 tunable-params twist while leaving the regime machine fully defined.
_ATR_WINDOW = 14
_RET_VOL_WINDOW = 20
_VOL_RANK_WINDOW = 252
_REGIME_SPLIT = 0.5      # vol percentile median: >= is stressed regime
_BREAKOUT_RANK = 0.80    # calm-regime momentum entry: price near top of range
_DIP_RANK = 0.20         # stressed-regime mean-reversion entry: spring compressed


@dataclass(slots=True)
class SpringRegimeParams:
    # Lookback for the price rolling percentile rank (the shared entry primitive).
    rank_window: int = 30
    # k in the k*ATR rolling-high trailing stop.
    trail_atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[SpringRegimeParams]):
    strategy_id = "gen_a1_1778885647"

    @classmethod
    def params_type(cls):
        return SpringRegimeParams

    def warmup_bars(self, params: SpringRegimeParams) -> int:
        rank_w = int(params.rank_window)
        # pct_change + ret-vol std + vol percentile rank chain is the longest path.
        vol_chain = _RET_VOL_WINDOW + _VOL_RANK_WINDOW + 1
        return int(max(rank_w + 1, vol_chain, _ATR_WINDOW + 1))

    def indicators(self, data: pd.DataFrame, params: SpringRegimeParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        # Average True Range (NaN during warmup, handled downstream).
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(_ATR_WINDOW).mean()

        # Price location within its own recent range: the shared entry primitive.
        rank_w = max(int(params.rank_window), 2)
        price_rank = close.rolling(rank_w).rank(pct=True)

        # Realized-volatility regime gate: percentile rank of short-window return std.
        ret = close.pct_change()
        realized_vol = ret.rolling(_RET_VOL_WINDOW).std()
        vol_rank = realized_vol.rolling(_VOL_RANK_WINDOW).rank(pct=True)

        return pd.DataFrame(
            {
                "atr": atr,
                "price_rank": price_rank,
                "vol_rank": vol_rank,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringRegimeParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        price_rank = indicators["price_rank"].to_numpy(dtype=float)
        vol_rank = indicators["vol_rank"].to_numpy(dtype=float)

        n = close.shape[0]
        pos = np.zeros(n, dtype=np.int64)
        k = float(params.trail_atr_mult)

        in_pos = False
        hwm = 0.0  # in-trade rolling-high water mark (only ratchets up)

        for i in range(n):
            pr = price_rank[i]
            vr = vol_rank[i]
            a = atr[i]

            if not in_pos:
                if np.isnan(pr) or np.isnan(vr) or np.isnan(a):
                    continue
                stressed = vr >= _REGIME_SPLIT
                if stressed:
                    # Compressed-spring regime: buy the over-stretched low.
                    entry = pr <= _DIP_RANK
                else:
                    # Calm regime: ride durable range-breakout strength.
                    entry = pr >= _BREAKOUT_RANK
                if entry:
                    in_pos = True
                    hwm = close[i]
                    pos[i] = 1
            else:
                c = close[i]
                if c > hwm:
                    hwm = c
                stop_dist = k * (a if not np.isnan(a) else 0.0)
                if c <= hwm - stop_dist:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(pos, index=data.index)
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
