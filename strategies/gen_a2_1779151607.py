from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# Fixed (non-tunable) constants. Keeping these out of the params class
# satisfies the <=2-tunable-params twist.
ATR_PERIOD = 14
ATR_RANK_LEN = 120          # lookback for the compression (ATR percentile) gate
COMPRESS_PCT = 0.25         # ATR rank below this => compressed / elastic regime
CHEAP_PCT = 0.35            # close rank below this => stretched-low inside the box
BE_TRIGGER = 0.03           # +3% reached => move stop to breakeven, then trail


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    trail_atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    """Elastic-recoil long: buy the first up-bar when the close is in its
    bottom percentile rank inside a compression-gated box; exit via a
    breakeven-then-trail ATR stop."""

    strategy_id = "gen_a2_1779151607"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(ATR_PERIOD + ATR_RANK_LEN, int(params.rank_window)) + 5

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

        rank_window = max(2, int(params.rank_window))
        # Vectorised rolling percentile rank of the last value in the window.
        atr_rank = atr.rolling(ATR_RANK_LEN, min_periods=ATR_RANK_LEN).rank(pct=True)
        close_rank = close.rolling(rank_window, min_periods=rank_window).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["atr_rank"] = atr_rank
        ind["close_rank"] = close_rank
        ind["ret"] = close.pct_change()
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        atr_rank = indicators["atr_rank"].to_numpy(dtype=float)
        close_rank = indicators["close_rank"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        n = close.shape[0]

        k = float(params.trail_atr_mult)

        # NaN-safe: comparisons against NaN evaluate to False, so warmup bars
        # never produce an entry.
        entry = (
            (atr_rank < COMPRESS_PCT)
            & (close_rank < CHEAP_PCT)
            & (ret > 0.0)
        )

        position = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        stop = 0.0
        be_done = False
        peak = 0.0

        # Path-dependent breakeven-then-trail exit: the stop never moves down.
        for i in range(n):
            if not in_pos:
                if bool(entry[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - k * atr[i]
                    be_done = False
                    peak = high[i]
                    position[i] = 1
            else:
                if high[i] > peak:
                    peak = high[i]
                # Phase 1: once +X% is reached, lift the stop to breakeven.
                if (not be_done) and high[i] >= entry_price * (1.0 + BE_TRIGGER):
                    if entry_price > stop:
                        stop = entry_price
                    be_done = True
                # Phase 2: after breakeven, trail by k*ATR off the running high.
                if be_done and np.isfinite(atr[i]):
                    cand = peak - k * atr[i]
                    if cand > stop:
                        stop = cand
                if low[i] <= stop:
                    in_pos = False
                    position[i] = 0
                else:
                    position[i] = 1

        sig = pd.Series(position, index=data.index)
        out = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decision on bar N's close fills on bar N+1.
        out["signal"] = sig.shift(1).fillna(0).astype(int)
        out["size"] = 1.0
        return SignalFrame(data=out, signal_column="signal", size_column="size")
