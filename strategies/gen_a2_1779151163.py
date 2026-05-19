from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed structural constants (intentionally not tunable to respect the <=2 param twist).
ATR_WINDOW = 14
MAX_HOLD = 2
CLV_THRESHOLD = 0.5


@dataclass(slots=True)
class Params:
    breakeven_pct: float = 0.01
    trail_atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779151163"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        # ATR needs ATR_WINDOW true ranges, true range needs one prior close.
        return ATR_WINDOW + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()

        prior_high = high.shift(1)
        prior_low = low.shift(1)
        # Outside bar: this bar's range engulfs the prior bar's range on both sides.
        outside = (high > prior_high) & (low < prior_low)
        outside = outside.fillna(False)

        rng = (high - low).to_numpy(dtype=float)
        safe_rng = np.where(rng > 0.0, rng, 1.0)
        clv_vals = np.where(
            rng > 0.0,
            (close - low).to_numpy(dtype=float) / safe_rng,
            0.5,
        )
        clv = pd.Series(clv_vals, index=data.index)

        raw = pd.Series(0, index=data.index, dtype="int64")
        raw[outside & (clv > CLV_THRESHOLD)] = 1
        raw[outside & (clv < CLV_THRESHOLD)] = -1

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["raw_signal"] = raw
        return out

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
        raw = indicators["raw_signal"].to_numpy(dtype=float)
        n = len(close)
        decision = np.zeros(n, dtype=np.int64)

        be = float(params.breakeven_pct)
        k = float(params.trail_atr_mult)

        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        bars_held = 0

        for i in range(n):
            just_exited = False

            if position != 0:
                bars_held += 1
                a = atr[i]
                if not np.isfinite(a):
                    a = 0.0

                if position == 1:
                    # Breakeven: once +be reached, ratchet stop up to entry.
                    if not armed and high[i] >= entry_price * (1.0 + be):
                        armed = True
                        if entry_price > stop:
                            stop = entry_price
                    # Trail: stop only ever moves up.
                    if armed:
                        cand = high[i] - k * a
                        if cand > stop:
                            stop = cand
                    if low[i] <= stop or bars_held >= MAX_HOLD:
                        position = 0
                        just_exited = True
                else:
                    # Short: breakeven ratchets stop down to entry.
                    if not armed and low[i] <= entry_price * (1.0 - be):
                        armed = True
                        if entry_price < stop:
                            stop = entry_price
                    # Trail: stop only ever moves down.
                    if armed:
                        cand = low[i] + k * a
                        if cand < stop:
                            stop = cand
                    if high[i] >= stop or bars_held >= MAX_HOLD:
                        position = 0
                        just_exited = True

                if just_exited:
                    armed = False
                    bars_held = 0

            if position == 0 and not just_exited:
                sig = raw[i]
                a = atr[i]
                if (sig == 1.0 or sig == -1.0) and np.isfinite(a) and a > 0.0:
                    position = int(sig)
                    entry_price = close[i]
                    armed = False
                    bars_held = 0
                    if position == 1:
                        stop = close[i] - k * a
                    else:
                        stop = close[i] + k * a

            decision[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decision
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
